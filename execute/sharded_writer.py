"""Crash-resilient, sharded, continuously-uploaded h5 writing for scripts/.

Every sweep script under scripts/ produces one logical output per layer that
can run into the hundreds of GB. Instead of one big h5 file per layer sitting
on NFS for the whole run, `ShardedLayerWriter` splits each layer's rows into
row-count shards, written to node-local scratch disk (never NFS -- NFS only
ever holds small progress/manifest JSON), and uploads each shard to the
remote store as it fills. A per-layer progress JSON on NFS records which
shards are fully uploaded, so a crash mid-run only loses whatever shard(s)
weren't yet confirmed uploaded -- on the next run, `ShardedLayerWriter`
deletes any such unfinished shard from local scratch and recomputes it from
scratch; every confirmed-complete shard is skipped.

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

from execute.remote_storage import AsyncPusher


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
class ShardedLayerWriter:
    """Manages one layer's sharded, checkpointed, continuously-uploaded h5
    output. `layer_key` is the project-relative path (no extension) this
    layer's shards and progress file live under, e.g.
    "onnx_models/smoll/tangent_manifold_distances/embedding.add_6".

    `schema` maps dataset name -> (per_row_shape, dtype), e.g.
    {"vec": ((768,), "float32"), "pos": ((), "int64")}. Every dataset shares
    the same leading (row) dimension.

    A shard only ever exists in one of two states as far as resuming is
    concerned: fully uploaded (recorded in `completed_shards`) or not
    started yet. There's no partial-shard resume -- on construction, any
    local scratch file for a shard that isn't in `completed_shards` is
    deleted and that shard is recomputed from its row_start, even if the
    previous run got most of the way through it. This trades a bit of
    wasted recompute after a crash for not having to reason about torn
    HDF5 writes, stale files from a differently-configured previous run, or
    progress.json/local-disk/remote-store states disagreeing with each
    other.
    """

    layer_key: str
    n_rows: int
    schema: dict[str, tuple[tuple[int, ...], str]]
    shard_rows: int = 100_000
    flush_every: int = 20
    attrs: dict = field(default_factory=dict)
    no_push: bool = False
    defer_push: bool = False
    """When True, a shard's upload isn't submitted as soon as it fills --
    every shard is kept on local scratch disk until this layer's compute
    loop finishes, then close() submits them all at once (still bounded by
    `max_in_flight` via `pusher`). Avoids per-shard upload latency competing
    with the compute loop, at the cost of local scratch holding a whole
    layer's shards at once, and losing every not-yet-uploaded shard (not
    just the in-progress one) if the job dies before close() runs -- only
    use this on nodes with enough local scratch headroom (e.g. mauao, not
    ruapehu -- see CLAUDE.md). Ignored if `no_push` is also set."""
    force_push: bool = False
    """When True, shard uploads overwrite the remote version unconditionally
    on a content conflict instead of prompting/deferring -- see
    execute.remote_storage.push. Set this when the caller already knows any
    existing remote content for this layer is stale (e.g. --overwrite-remote
    on a script that just regenerated its output from scratch)."""
    auto_delete: bool = False
    """When False, a shard's local scratch file is kept after its upload is
    confirmed, instead of being deleted -- see execute.remote_storage.push.
    Ignored if `no_push` is set (nothing is uploaded, so nothing to delete
    either way)."""
    project_root: str = "."
    pusher: AsyncPusher | None = None
    """Shared AsyncPusher to upload shards through. `pusher` is typically
    shared across every layer's writer in a run, so different layers'
    uploads run concurrently even though each layer is capped individually
    by `max_in_flight`. When None (default) and `no_push` is False, this
    writer creates its own single-worker AsyncPusher."""
    max_in_flight: int = 1
    """Max number of this writer's own shard uploads allowed outstanding at
    once. Starting a new shard beyond this cap first waits for the oldest
    pending one to finish. Raise this if the sweep spends a lot of time
    blocked waiting on uploads instead of computing -- but each additional
    in-flight shard is another one of this layer's shard files sitting on
    local scratch disk until its upload completes, and another concurrent
    SFTP connection when many layers' writers hit this at once, so weigh it
    against local scratch headroom and the remote store's connection limits
    (see CLAUDE.md's SFTP timeout guidance). Ignored under --defer-push
    (every shard waits until close() regardless)."""

    def __post_init__(self) -> None:
        self._pending_uploads: list = []  # [(Future, shard dict), ...], oldest first
        self._owns_pusher = False
        if self.pusher is None and not self.no_push:
            self.pusher = AsyncPusher(max_workers=self.max_in_flight)
            self._owns_pusher = True

        self.progress_path = _progress_path(self.project_root, self.layer_key)
        self._progress = self._load_progress()
        self._scratch_dir = os.path.join(_scratch_root(), self.layer_key)
        os.makedirs(self._scratch_dir, exist_ok=True)

        self._cur_h5: h5py.File | None = None
        self._cur_shard_idx = len(self._progress["completed_shards"])
        self._cur_shard_start = (
            self._progress["completed_shards"][-1]["row_end"]
            if self._progress["completed_shards"] else 0
        )
        self.next_row = self._cur_shard_start
        self._cur_shard_path: str | None = None
        self._blocks_since_flush = 0

        self._discard_unfinished_shards()

        if self.next_row >= self.n_rows:
            print(f"[sharded_writer] {self.layer_key}: already complete "
                  f"({self.next_row}/{self.n_rows} rows) per {self.progress_path}")

    def _discard_unfinished_shards(self) -> None:
        """Delete every local scratch shard file at or past _cur_shard_idx --
        i.e. anything not backed by a completed_shards entry. Local scratch
        on a SLURM node persists across jobs (it's not scoped to a run), so
        this also cleans up leftovers from an earlier, differently
        configured attempt at this same layer_key, not just this run's own
        in-progress shard."""
        for path in sorted(glob.glob(os.path.join(self._scratch_dir, "shard_*.h5"))):
            idx = int(os.path.basename(path)[len("shard_"):-len(".h5")])
            if idx >= self._cur_shard_idx:
                print(f"[sharded_writer] {self.layer_key}: discarding unfinished "
                      f"shard file {path} (will be recomputed)")
                os.remove(path)

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
        self._progress["next_row"] = self.next_row
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

    def _wait_oldest_pending_upload(self) -> None:
        """Block until this writer's single oldest pending upload finishes,
        then mark its completed_shards entry as pushed. No-op if nothing is
        pending."""
        if not self._pending_uploads:
            return
        future, shard_entry = self._pending_uploads.pop(0)
        future.result()  # raises if the upload failed -- surface it here, not silently
        print(f"[sharded_writer] {self.layer_key}: shard {shard_entry['shard_idx']} "
              f"(rows {shard_entry['row_start']}-{shard_entry['row_end']}) pushed to remote"
              + (" and deleted locally" if self.auto_delete else " (kept locally)"))
        shard_entry["pushed"] = True
        self._save_progress()

    def _wait_pending_uploads(self, keep: int = 0) -> None:
        """Drain oldest-first until at most `keep` uploads remain pending.
        Called before starting a new shard's upload (caps this writer at
        `max_in_flight` shards ahead of what's confirmed pushed, `keep =
        max_in_flight - 1`) and from close() with `keep=0` (nothing left to
        overlap the final upload with)."""
        while len(self._pending_uploads) > keep:
            self._wait_oldest_pending_upload()

    def _submit_upload(self, local_path: str, shard_entry: dict) -> None:
        self._wait_pending_uploads(keep=self.max_in_flight - 1)
        future = self.pusher.submit(
            local_path, virtual_path=shard_entry["virtual_path"], force=self.force_push,
            auto_delete=self.auto_delete,
        )
        self._pending_uploads.append((future, shard_entry))
        print(f"[sharded_writer] {self.layer_key}: shard {shard_entry['shard_idx']} "
              f"(rows {shard_entry['row_start']}-{shard_entry['row_end']}) queued for upload")

    def _close_and_upload_shard(self) -> None:
        assert self._cur_h5 is not None and self._cur_shard_path is not None
        self._cur_h5.attrs["row_end"] = self.next_row
        self._cur_h5.close()

        shard_entry = {
            "shard_idx": self._cur_shard_idx,
            "row_start": self._cur_shard_start,
            "row_end": self.next_row,
            "virtual_path": _shard_virtual_path(self.layer_key, self._cur_shard_idx),
        }

        shard_entry["pushed"] = False  # set True once the upload (if any) is confirmed
        self._progress["completed_shards"].append(shard_entry)
        self._save_progress()

        if self.no_push:
            print(f"[sharded_writer] {self.layer_key}: shard {shard_entry['shard_idx']} "
                  f"(rows {shard_entry['row_start']}-{shard_entry['row_end']}) kept locally at "
                  f"{self._cur_shard_path} (--no-push)")
        elif not self.defer_push:
            self._submit_upload(self._cur_shard_path, shard_entry)
        # else: --defer-push -- left on local scratch, uploaded from close()

        self._cur_h5 = None
        self._cur_shard_path = None
        self._cur_shard_idx += 1
        self._cur_shard_start = self.next_row

    def _push_deferred_shards(self) -> None:
        """Upload every completed_shards entry not yet pushed (i.e. every
        shard written this run under --defer-push), in order, bounded by
        max_in_flight. Called from close() once this layer's compute loop is
        entirely done."""
        for shard_entry in self._progress["completed_shards"]:
            if shard_entry["pushed"]:
                continue
            local_path = os.path.join(
                self._scratch_dir, f"shard_{shard_entry['shard_idx']:05d}.h5"
            )
            self._submit_upload(local_path, shard_entry)

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

        if self.defer_push and not self.no_push:
            self._push_deferred_shards()

        self._wait_pending_uploads(keep=0)

        write_manifest(self.project_root, self.layer_key, self._progress)

        if self._owns_pusher:
            self.pusher.shutdown()

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
    pull_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """Loader-side helper: read a layer's manifest, `ensure_local` (pull from
    remote store, if not already local) every shard, and concatenate each
    requested dataset across shards in row order. For layers whose shards
    were kept locally (--no-push), reads them directly from scratch instead.

    `pull_dir`: pull missing shards into this directory instead of under
    `project_root` (NFS) -- e.g. node-local scratch disk, for a caller that
    only needs the rows transiently and doesn't want to grow shared NFS
    storage for a full multi-GB shard. Checked for an already-pulled copy
    before re-pulling, same as the NFS-default path via ensure_local's own
    existence check. Defaults to None (the original NFS-under-project_root
    behavior, unchanged)."""
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
            local_path = ensure_local(shard["virtual_path"], dest_dir=pull_dir)
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
