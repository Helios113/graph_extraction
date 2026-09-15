"""Crash-resilient, sharded, continuously-uploaded h5 writing for scripts/.

Every sweep script under scripts/ produces one logical output per layer that
can run into the hundreds of GB. Instead of one big h5 file per layer sitting
on NFS for the whole run, `ShardedLayerWriter` splits each layer's rows into
row-count shards, written to node-local scratch disk (never NFS -- NFS only
ever holds small progress/manifest JSON), and uploads+deletes each shard as
soon as it's full. A per-layer progress JSON on NFS records which shards are
already done, so a crash mid-run only loses the in-progress shard's rows
(still on local disk, not yet uploaded) -- re-running the same command skips
every completed shard and resumes from its `next_row`.

Typical use (see scripts/generate_tangent_manifold_distances.py for the full
integration):

    writer = ShardedLayerWriter(
        layer_key="onnx_models/smoll/tangent_manifold_distances/embedding.add_6",
        n_rows=n_rows,
        schema={
            "input_distance": ((n_samples,), "float32"),
            "output_distance": ((n_samples,), "float32"),
            "k": ((), "int64"),
            ...
        },
        shard_rows=args.shard_rows,
        flush_every=args.flush_every,
        attrs={"n_samples": args.n_samples, "radius": args.radius},
        no_push=args.no_push,
    )
    for block_start in range(writer.next_row, n_rows, rows_per_block):
        ...compute block...
        writer.write_rows(block_start, block_end, input_distance=..., k=..., ...)
    writer.close()

`writer.next_row` is where to resume from after a crash -- callers should
skip directly to it rather than starting the loop at 0.
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import time
from dataclasses import dataclass, field

import h5py
import numpy as np

from execute.remote_storage import AsyncPusher, push


def _scratch_root() -> str:
    """Node-local scratch dir for in-progress shard writes -- never NFS.
    Override with $NEW_JAC_SCRATCH (e.g. if /local/scratch isn't the right
    disk on a given node)."""
    root = os.environ.get("NEW_JAC_SCRATCH")
    if root:
        return root
    return os.path.join("/local/scratch", getpass.getuser(), "new_jac")


def _progress_path(project_root: str, layer_key: str) -> str:
    return os.path.join(project_root, layer_key + ".progress.json")


def _shard_virtual_path(layer_key: str, shard_idx: int) -> str:
    return f"{layer_key}/shard_{shard_idx:05d}.h5"


@dataclass
class _CompletedShard:
    shard_idx: int
    row_start: int
    row_end: int
    virtual_path: str


@dataclass
class ShardedLayerWriter:
    """Manages one layer's sharded, checkpointed, continuously-uploaded h5
    output. `layer_key` is the project-relative path (no extension) this
    layer's shards and progress file live under, e.g.
    "onnx_models/smoll/tangent_manifold_distances/embedding.add_6".

    `schema` maps dataset name -> (per_row_shape, dtype), e.g.
    {"vec": ((768,), "float32"), "pos": ((), "int64")}. Every dataset shares
    the same leading (row) dimension.
    """

    layer_key: str
    n_rows: int
    schema: dict[str, tuple[tuple[int, ...], str]]
    shard_rows: int = 100_000
    flush_every: int = 20
    attrs: dict = field(default_factory=dict)
    no_push: bool = False
    force_push: bool = False
    """When True, shard uploads overwrite the remote version unconditionally
    on a content conflict instead of prompting/deferring -- see
    execute.remote_storage.push. Set this when the caller already knows any
    existing remote content for this layer is stale (e.g. --overwrite-remote
    on a script that just regenerated its output from scratch)."""
    project_root: str = "."
    pusher: AsyncPusher | None = None
    """Shared AsyncPusher to upload shards through without blocking the
    caller's compute loop. When set, at most one of this writer's shards may
    be uploading at a time -- starting a new shard first waits for the
    previous one's upload to finish, so this writer never gets more than one
    shard ahead of what's been pushed. `pusher` is typically shared across
    every layer's writer in a run, so different layers' uploads still run
    concurrently even though each layer is capped at one in flight. When
    None (default), falls back to the old behavior: push() blocks inline."""

    def __post_init__(self) -> None:
        self._pending_upload = None  # (Future, shard dict) for this writer's one in-flight upload
        self.progress_path = _progress_path(self.project_root, self.layer_key)
        self._progress = self._load_progress()
        self.next_row = self._progress["next_row"]
        self._scratch_dir = os.path.join(_scratch_root(), self.layer_key)
        os.makedirs(self._scratch_dir, exist_ok=True)

        self._cur_h5: h5py.File | None = None
        self._cur_shard_idx = len(self._progress["completed_shards"])
        self._cur_shard_start = (
            self._progress["completed_shards"][-1]["row_end"]
            if self._progress["completed_shards"] else 0
        )
        self._cur_shard_path: str | None = None
        self._blocks_since_flush = 0

        if self.next_row >= self.n_rows:
            # Progress says every shard is written and pushed -- but that's
            # just what a prior run's bookkeeping claims, not a live fact.
            # If a push was interrupted after upload but before (or without)
            # its pointer landing on the remote index, or the remote store
            # was reset/rebuilt after the push happened, completed_shards
            # can say "pushed": true for a shard nothing durable actually
            # backs -- both local scratch (already deleted, since push()
            # auto-deletes on success) and the remote index are empty for
            # it. Confirm each pushed shard's pointer is still present
            # before trusting "complete"; demote (and roll next_row back to)
            # the earliest one that isn't, so the caller recomputes it and
            # everything after it, the same as if it had never finished.
            self._verify_pushed_shards_or_rollback()

        if self.next_row >= self.n_rows:
            print(f"[sharded_writer] {self.layer_key}: already complete "
                  f"({self.next_row}/{self.n_rows} rows) per {self.progress_path}")
            return

        # If progress says we're partway through a shard (next_row is past
        # this shard's start but the shard itself isn't marked done), the
        # previous run's partial shard file should still be on this node's
        # local scratch disk (jobs are pinned to one SLURM node -- see
        # CLAUDE.md) -- reopen and keep appending to it rather than
        # recomputing rows we already have. If that file is missing, or
        # present but corrupt (e.g. the previous run was killed mid-write,
        # before an in-progress HDF5 header/block update landed -- a SIGKILL
        # or node failure doesn't respect h5py's internal write ordering), we
        # can't trust next_row for this partial shard: roll back to the
        # shard's start and let the caller recompute it -- the
        # completed_shards list (and therefore _cur_shard_start) only
        # advances once a shard is fully closed and its progress entry
        # written, so this only ever re-does the one in-flight shard, never a
        # finished one.
        candidate_path = os.path.join(
            self._scratch_dir, f"shard_{self._cur_shard_idx:05d}.h5"
        )
        candidate_usable = os.path.exists(candidate_path)
        if candidate_usable:
            try:
                with h5py.File(candidate_path, "r"):
                    pass
            except OSError as e:
                print(f"[sharded_writer] {self.layer_key}: in-progress shard "
                      f"{self._cur_shard_idx} on local scratch disk "
                      f"({candidate_path}) is corrupt ({e}) -- discarding it")
                os.remove(candidate_path)
                candidate_usable = False

        if self.next_row > self._cur_shard_start and not candidate_usable:
            print(f"[sharded_writer] {self.layer_key}: in-progress shard "
                  f"{self._cur_shard_idx} not found on local scratch disk "
                  f"({candidate_path}) -- rolling back to re-process it from "
                  f"row {self._cur_shard_start}")
            self.next_row = self._cur_shard_start
            self._save_progress()

    def _verify_pushed_shards_or_rollback(self) -> None:
        """Check each completed_shards entry marked "pushed": true actually
        has a pointer in the remote index. Local scratch can't help here --
        push() deletes the local shard file on success, so by the time
        completed_shards says a shard is done, its only copy is supposed to
        be remote. On the first entry that isn't actually there, drop it and
        every later entry (they're written in order, so a later shard being
        fine doesn't mean anything if an earlier one has to be
        recomputed -- row ranges are contiguous) and roll next_row back to
        where the dropped entry started, so the caller regenerates it (and
        everything after) as if it had never been pushed."""
        from execute.remote_storage import remote_exists

        shards = self._progress["completed_shards"]
        for i, shard in enumerate(shards):
            if not shard["pushed"]:
                continue
            if remote_exists(shard["virtual_path"]):
                continue
            print(f"[sharded_writer] {self.layer_key}: shard {shard['shard_idx']} "
                  f"(rows {shard['row_start']}-{shard['row_end']}) is marked pushed "
                  f"but has no pointer at remote index for {shard['virtual_path']!r} "
                  "-- treating as not pushed and rolling back to regenerate it "
                  f"(and any shards after it) from row {shard['row_start']}")
            self._progress["completed_shards"] = shards[:i]
            self.next_row = shard["row_start"]
            self._cur_shard_idx = i
            self._cur_shard_start = shard["row_start"]
            self._save_progress()
            return

    # -- progress bookkeeping -------------------------------------------------

    def _load_progress(self) -> dict:
        if os.path.exists(self.progress_path):
            with open(self.progress_path) as f:
                data = json.load(f)
            print(f"[sharded_writer] {self.layer_key}: resuming from row "
                  f"{data['next_row']}/{self.n_rows} "
                  f"({len(data['completed_shards'])} shard(s) already done)")
            return data
        return {
            "layer_key": self.layer_key,
            "n_rows": self.n_rows,
            "shard_rows": self.shard_rows,
            "schema": {k: [list(shape), dtype] for k, (shape, dtype) in self.schema.items()},
            "attrs": self.attrs,
            "completed_shards": [],
            "next_row": 0,
            "updated_at": None,
        }

    def _save_progress(self) -> None:
        # If the previous shard's upload hasn't been confirmed yet (it's
        # still in self._pending_upload), completed_shards doesn't include
        # it -- so _cur_shard_start is where that unconfirmed shard ends /
        # the new shard begins. Persisting self.next_row here, if it's
        # already past that point, would claim rows in the new shard are
        # safe to resume from while the shard before them is unrecorded --
        # exactly the torn state (next_row far ahead of completed_shards)
        # that breaks resume if the pending upload later fails. Cap what's
        # written at _cur_shard_start until _wait_pending_upload() confirms
        # it and this cap no longer applies.
        safe_next_row = self.next_row
        if self._pending_upload is not None:
            safe_next_row = min(safe_next_row, self._cur_shard_start)
        self._progress["next_row"] = safe_next_row
        self._progress["updated_at"] = time.time()
        os.makedirs(os.path.dirname(self.progress_path), exist_ok=True)
        tmp_path = self.progress_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._progress, f, indent=2)
        os.replace(tmp_path, self.progress_path)  # atomic on the same filesystem

    # -- shard lifecycle -------------------------------------------------------

    def _open_shard(self) -> None:
        capacity = min(self.shard_rows, self.n_rows - self._cur_shard_start)
        self._cur_shard_path = os.path.join(
            self._scratch_dir, f"shard_{self._cur_shard_idx:05d}.h5"
        )
        if os.path.exists(self._cur_shard_path):
            # Resuming mid-shard after a crash -- the partial file from the
            # previous run is still here (same node/scratch disk); reopen
            # for read/write instead of truncating, so the rows it already
            # has survive. __post_init__ already validated this file opens
            # cleanly and discarded it otherwise, so this should always
            # succeed -- but guard anyway rather than crashing the whole
            # sweep on any surprise corruption (e.g. a concurrent process
            # touching the same scratch path).
            try:
                h5f = h5py.File(self._cur_shard_path, "r+")
                assert h5f.attrs["row_start"] == self._cur_shard_start
            except OSError as e:
                raise OSError(
                    f"{self.layer_key}: shard {self._cur_shard_idx} at "
                    f"{self._cur_shard_path} is corrupt ({e}) even though it "
                    "passed the startup check -- delete it manually and "
                    "re-run to recompute this shard from row "
                    f"{self._cur_shard_start}"
                ) from e
        else:
            h5f = h5py.File(self._cur_shard_path, "w")
            h5f.attrs["row_start"] = self._cur_shard_start
            h5f.attrs["row_end"] = self._cur_shard_start + capacity  # provisional, fixed on close
            for k, v in self.attrs.items():
                h5f.attrs[k] = v
            for name, (shape, dtype) in self.schema.items():
                h5f.create_dataset(name, shape=(capacity, *shape), dtype=dtype)
        self._cur_h5 = h5f

    def write_rows(self, row_start: int, row_end: int, **datasets: np.ndarray) -> None:
        """Write rows [row_start, row_end) of one or more datasets (matching
        `schema` keys). Rows must be written in increasing order and must
        fall within the writer's current shard (i.e. callers should derive
        their block boundaries from `writer.next_row` / shard_rows, or simply
        call this once per block -- shard rollover is handled here)."""
        if row_end <= row_start:
            return
        if row_start < self.next_row:
            raise ValueError(
                f"row_start={row_start} is before next_row={self.next_row} -- "
                "resume from writer.next_row, don't re-process already-written rows"
            )
        if row_start != self.next_row:
            raise ValueError(f"expected row_start == next_row ({self.next_row}), got {row_start}")

        pos = row_start
        while pos < row_end:
            if self._cur_h5 is None:
                self._open_shard()
            shard_capacity_end = self._cur_shard_start + min(
                self.shard_rows, self.n_rows - self._cur_shard_start
            )
            chunk_end = min(row_end, shard_capacity_end)
            local_start = pos - self._cur_shard_start
            local_end = chunk_end - self._cur_shard_start

            for name, arr in datasets.items():
                src_start = pos - row_start
                src_end = src_start + (chunk_end - pos)
                self._cur_h5[name][local_start:local_end] = arr[src_start:src_end]

            pos = chunk_end
            self.next_row = pos

            self._blocks_since_flush += 1
            is_shard_full = pos == shard_capacity_end
            if is_shard_full or self._blocks_since_flush >= self.flush_every:
                self._cur_h5.flush()
                self._blocks_since_flush = 0

            if is_shard_full:
                self._close_and_upload_shard()

        self._save_progress()

    def _wait_pending_upload(self) -> None:
        """Block until this writer's one allowed in-flight upload (if any)
        finishes, then record it as completed in progress/manifest. Called
        before starting a new shard (caps this writer at one shard ahead of
        what's confirmed pushed) and from close() (nothing left to overlap
        the final upload with)."""
        if self._pending_upload is None:
            return
        future, shard_entry = self._pending_upload
        future.result()  # raises if the upload failed -- surface it here, not silently
        print(f"[sharded_writer] {self.layer_key}: shard {shard_entry['shard_idx']} "
              f"(rows {shard_entry['row_start']}-{shard_entry['row_end']}) "
              "pushed to remote and deleted locally")
        self._progress["completed_shards"].append(shard_entry)
        self._save_progress()
        self._pending_upload = None

    def _close_and_upload_shard(self) -> None:
        assert self._cur_h5 is not None and self._cur_shard_path is not None
        self._cur_h5.attrs["row_end"] = self.next_row
        self._cur_h5.close()

        virtual_path = _shard_virtual_path(self.layer_key, self._cur_shard_idx)
        shard = _CompletedShard(
            shard_idx=self._cur_shard_idx,
            row_start=self._cur_shard_start,
            row_end=self.next_row,
            virtual_path=virtual_path,
        )
        shard_entry = {
            "shard_idx": shard.shard_idx,
            "row_start": shard.row_start,
            "row_end": shard.row_end,
            "virtual_path": shard.virtual_path,
            "pushed": not self.no_push,
        }

        if self.no_push:
            print(f"[sharded_writer] {self.layer_key}: shard {self._cur_shard_idx} "
                  f"(rows {shard.row_start}-{shard.row_end}) kept locally at {self._cur_shard_path} "
                  "(--no-push)")
            self._progress["completed_shards"].append(shard_entry)
            self._save_progress()
        elif self.pusher is not None:
            # Wait for this writer's *previous* upload (if still running)
            # before kicking off this one, so it never has more than one
            # shard's worth of unpushed data in flight -- the compute loop
            # keeps going in the meantime via the shared process pool.
            self._wait_pending_upload()
            future = self.pusher.submit(
                self._cur_shard_path, virtual_path=virtual_path, force=self.force_push,
            )
            self._pending_upload = (future, shard_entry)
            print(f"[sharded_writer] {self.layer_key}: shard {self._cur_shard_idx} "
                  f"(rows {shard.row_start}-{shard.row_end}) queued for upload")
        else:
            # auto_delete=True removes the local file on success
            push(self._cur_shard_path, virtual_path=virtual_path, force=self.force_push)
            print(f"[sharded_writer] {self.layer_key}: shard {self._cur_shard_idx} "
                  f"(rows {shard.row_start}-{shard.row_end}) pushed to remote and deleted locally")
            self._progress["completed_shards"].append(shard_entry)
            self._save_progress()

        self._cur_h5 = None
        self._cur_shard_path = None
        self._cur_shard_idx += 1
        self._cur_shard_start = self.next_row

    def close(self) -> None:
        """Flush/close+upload any partially-filled final shard (a shard
        doesn't have to be exactly shard_rows to be finished -- the last
        shard of a layer usually isn't)."""
        if self._cur_h5 is not None:
            capacity = self._cur_h5.attrs["row_end"] - self._cur_h5.attrs["row_start"]
            written = self.next_row - self._cur_shard_start
            if written < capacity:
                for name in self.schema:
                    dset = self._cur_h5[name]
                    new_shape = (written, *dset.shape[1:])
                    dset.resize(new_shape)
            self._close_and_upload_shard()

        self._wait_pending_upload()
        write_manifest(self.project_root, self.layer_key, self._progress)

        try:
            os.rmdir(self._scratch_dir)  # only succeeds if empty (all shards pushed+deleted)
        except OSError:
            pass

        if self.next_row >= self.n_rows:
            print(f"[sharded_writer] {self.layer_key}: complete "
                  f"({self.next_row}/{self.n_rows} rows, "
                  f"{len(self._progress['completed_shards'])} shard(s))")
        else:
            print(f"[sharded_writer] {self.layer_key}: stopped at "
                  f"{self.next_row}/{self.n_rows} rows -- re-run the same command to resume")


def write_manifest(project_root: str, layer_key: str, progress: dict) -> None:
    """Refresh the human-browsable NFS manifest for one layer (a copy of its
    progress, plus whether it's complete) at
    <layer_key>.manifest.json -- e.g. loaders can check this (or the
    .progress.json directly) to see what shards exist for a layer and where
    they are, without touching the remote store."""
    manifest_path = os.path.join(project_root, layer_key + ".manifest.json")
    manifest = {
        "layer_key": layer_key,
        "n_rows": progress["n_rows"],
        "shard_rows": progress["shard_rows"],
        "schema": progress["schema"],
        "attrs": progress["attrs"],
        "complete": progress["next_row"] >= progress["n_rows"],
        "next_row": progress["next_row"],
        "shards": progress["completed_shards"],
        "updated_at": progress["updated_at"],
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


def load_shard_rows(
    project_root: str,
    layer_key: str,
    dataset_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Loader-side helper: read a layer's manifest, `ensure_local` (pull from
    remote store, if not already local) every shard, and concatenate each
    requested dataset across shards in row order. For layers whose shards
    were kept locally (--no-push), reads them directly from scratch instead.
    """
    from execute.remote_storage import ensure_local

    manifest_path = os.path.join(project_root, layer_key + ".manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest["complete"]:
        print(f"[sharded_writer] warning: {layer_key} manifest is not marked complete "
              f"({manifest['next_row']}/{manifest['n_rows']} rows)")

    names = dataset_names or list(manifest["schema"])
    out: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for shard in sorted(manifest["shards"], key=lambda s: s["shard_idx"]):
        if shard["pushed"]:
            ensure_local(shard["virtual_path"])  # relative to PROJECT_ROOT, resolved there
            local_path = os.path.join(project_root, shard["virtual_path"])
        else:
            local_path = os.path.join(
                _scratch_root(), layer_key, f"shard_{shard['shard_idx']:05d}.h5"
            )
        with h5py.File(local_path, "r") as h5f:
            for name in names:
                out[name].append(h5f[name][:])

    return {name: np.concatenate(chunks, axis=0) for name, chunks in out.items()}


def discover_layer_names(layers_dir: str) -> list[str]:
    """List the layer names under `layers_dir` that have a ShardedLayerWriter
    manifest (i.e. <layers_dir>/<layer_name>.manifest.json)."""
    manifests = sorted(glob.glob(os.path.join(layers_dir, "*.manifest.json")))
    return [os.path.basename(p)[: -len(".manifest.json")] for p in manifests]


def load_base_seqs(seqs_key: str) -> np.ndarray:
    """Reads the full input_ids array from the sharded output written by
    scripts.generate_base_sequences.tokenize_and_save_sequences (layer_key
    == seqs_key, single "input_ids" dataset)."""
    return load_shard_rows(".", seqs_key, dataset_names=["input_ids"])["input_ids"]


def load_activations(activations_dir: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reads base_seqs.h5 plus every layer's sharded "activation" dataset
    (bf16 stored as raw uint16, viewed back here) under
    activations_dir/activations/ -- the format written by
    scripts.generate_base_activations.collect_and_save_activations. Shared by
    every script that consumes base activations (compute_base_jacobian_stats,
    generate_tangent_manifold_distances)."""
    import ml_dtypes

    with h5py.File(os.path.join(activations_dir, "base_seqs.h5"), "r") as h5f:
        base_seqs = h5f["base_seqs"][:]

    layers_dir = os.path.join(activations_dir, "activations")
    activations = {
        name: load_shard_rows(".", os.path.join(layers_dir, name), dataset_names=["activation"])[
            "activation"
        ].view(ml_dtypes.bfloat16)
        for name in discover_layer_names(layers_dir)
    }
    return base_seqs, activations
