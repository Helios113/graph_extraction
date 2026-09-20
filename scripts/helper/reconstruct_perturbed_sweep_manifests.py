"""Reconstructs progress.json + manifest for layers whose perturbed-sweep
shards exist on local scratch but have NEITHER a progress.json NOR a
manifest -- e.g. a sweep job that was killed hard enough that even
_save_progress() never ran (unlike scripts.helper.finish_perturbed_sweep_manifests,
which assumes progress.json survived and only the final close()/manifest step
was lost).

Every shard written by execute.sharded_writer.ShardedLayerWriter is
self-describing (attrs: d_model, n_samples, seq, row_start, row_end, plus the
dataset shapes/dtypes themselves), so a layer's full progress dict can be
rebuilt purely by reading its shard_*.h5 files off disk, validating that
row_start/row_end are contiguous from 0 (catching a missing/corrupt shard
before anything is pushed), and inferring shard_rows from the first
non-final shard's row count and n_rows from the last shard's row_end.

Does the same two things finish_perturbed_sweep_manifests.py does once the
progress dict is in hand, but as two full passes over every layer, manifests
FIRST: write_manifest only depends on row counts (progress["next_row"] >=
progress["n_rows"]), not push status, so every layer's progress.json +
manifest is written in pass 1 before pass 2 pushes any layer's shards --
a load_shard_rows caller on the same node still finds each shard locally via
its scratch fallback (shard["pushed"] == False), so this doesn't require the
pushes to have finished. This means the whole folder's manifests exist
quickly (pass 1 only reads shard headers, no uploads) instead of waiting on
one layer's full push queue (often 1000+s/shard) before the next layer's
manifest is even written. Each layer's manifest is rewritten (via
write_manifest) after each of its shards' push completes in pass 2 to update
its `pushed` flag, so a crash mid-push still leaves a valid, readable
manifest -- only "readable from the remote store on another node" degrades
until the interrupted pushes are retried, not "readable at all".

Run on the node that actually holds the local scratch data (e.g. via `srun
--partition=ampere --nodelist=mauao`), not the NFS login node.
"""

import argparse
import glob
import json
import os
import time

import h5py

from execute.remote_storage import push
from execute.sharded_writer import _scratch_root, _shard_virtual_path, write_manifest


def _discover_layers(activations_dir: str) -> list[str]:
    """Every subdirectory of `activations_dir` (on local scratch) containing
    at least one shard_*.h5 file, i.e. every layer with raw shard data --
    regardless of whether it has a progress.json or manifest."""
    scratch_dir = os.path.join(_scratch_root(), activations_dir)
    if not os.path.isdir(scratch_dir):
        return []
    layers = []
    for name in sorted(os.listdir(scratch_dir)):
        layer_dir = os.path.join(scratch_dir, name)
        if os.path.isdir(layer_dir) and glob.glob(os.path.join(layer_dir, "shard_*.h5")):
            layers.append(name)
    return layers


def _reconstruct_progress(activations_dir: str, layer: str) -> dict:
    """Reads every shard_*.h5 for `layer` off local scratch and rebuilds the
    same progress dict shape ShardedLayerWriter would have produced."""
    layer_key = os.path.join(activations_dir, layer)
    layer_dir = os.path.join(_scratch_root(), layer_key)
    shard_paths = sorted(glob.glob(os.path.join(layer_dir, "shard_*.h5")))
    assert shard_paths, f"no shard_*.h5 files found under {layer_dir}"

    completed_shards = []
    schema = None
    attrs = None
    expected_row_start = 0
    shard_rows = None

    for shard_path in shard_paths:
        shard_idx = int(os.path.basename(shard_path)[len("shard_"):-len(".h5")])
        with h5py.File(shard_path, "r") as f:
            row_start, row_end = int(f.attrs["row_start"]), int(f.attrs["row_end"])
            n_rows_here = row_end - row_start
            this_schema = {k: [list(f[k].shape[1:]), str(f[k].dtype)] for k in f.keys()}
            this_attrs = {k: (int(v) if hasattr(v, "item") else v) for k, v in f.attrs.items()
                          if k not in ("row_start", "row_end")}

        assert row_start == expected_row_start, (
            f"{layer}: shard {shard_idx} row_start={row_start} != expected {expected_row_start} "
            "-- a shard is missing or out of order, refusing to reconstruct"
        )
        if schema is None:
            schema, attrs = this_schema, this_attrs
        else:
            assert this_schema == schema, f"{layer}: shard {shard_idx} schema mismatch"
            assert this_attrs == attrs, f"{layer}: shard {shard_idx} attrs mismatch"

        # shard_rows = the configured max rows/shard, inferred from any non-final
        # (i.e. full-capacity) shard; the last shard may be smaller (partial).
        if shard_rows is None or n_rows_here > shard_rows:
            shard_rows = n_rows_here

        completed_shards.append(dict(
            shard_idx=shard_idx,
            row_start=row_start,
            row_end=row_end,
            virtual_path=_shard_virtual_path(layer_key, shard_idx),
            pushed=False,
        ))
        expected_row_start = row_end

    return {
        "layer_key": layer_key,
        "n_rows": expected_row_start,
        "shard_rows": shard_rows,
        "schema": schema,
        "attrs": attrs,
        "completed_shards": completed_shards,
        "next_row": expected_row_start,
        "updated_at": time.time(),
    }


def write_manifest_for_layer(activations_dir: str, layer: str, force: bool) -> bool:
    """Pass 1: reads layer's shards off local scratch and writes
    progress.json + manifest.json (every shard's `pushed` still False).
    Returns True if this layer needs pass 2 (push_layer_shards), False if it
    was skipped (already had progress.json/manifest and not --force)."""
    layer_key = os.path.join(activations_dir, layer)
    progress_path = layer_key + ".progress.json"
    manifest_path = layer_key + ".manifest.json"

    if (os.path.exists(progress_path) or os.path.exists(manifest_path)) and not force:
        print(f"[{layer}] skipping: progress.json or manifest already exists (pass --force to rewrite)", flush=True)
        return False

    print(f"[{layer}] reading shards from local scratch...", flush=True)
    progress = _reconstruct_progress(activations_dir, layer)
    print(f"[{layer}] reconstructed: {len(progress['completed_shards'])} shard(s), "
          f"{progress['n_rows']} rows, shard_rows={progress['shard_rows']}", flush=True)

    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)
    print(f"[{layer}] progress.json written -> {progress_path}", flush=True)

    write_manifest(".", layer_key, progress)
    print(f"[{layer}] manifest written -> {manifest_path}", flush=True)
    return True


def push_layer_shards(activations_dir: str, layer: str) -> None:
    """Pass 2: pushes every not-yet-pushed shard in layer's progress.json
    (written by pass 1, or already on disk from a previous run) and
    rewrites the manifest after each one to keep its `pushed` flags current."""
    layer_key = os.path.join(activations_dir, layer)
    progress_path = layer_key + ".progress.json"
    manifest_path = layer_key + ".manifest.json"

    with open(progress_path) as f:
        progress = json.load(f)

    unpushed = [s for s in progress["completed_shards"] if not s["pushed"]]
    if not unpushed:
        print(f"[{layer}] all shards already pushed", flush=True)
        return

    for shard in unpushed:
        scratch_path = os.path.join(_scratch_root(), layer_key, f"shard_{shard['shard_idx']:05d}.h5")
        t0 = time.time()
        print(f"[{layer}] pushing shard {shard['shard_idx']} ({scratch_path})...", flush=True)
        push(scratch_path, virtual_path=shard["virtual_path"])
        shard["pushed"] = True
        progress["updated_at"] = time.time()
        tmp_path = progress_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp_path, progress_path)
        write_manifest(".", layer_key, progress)
        print(f"[{layer}] shard {shard['shard_idx']} pushed in {time.time() - t0:.1f}s", flush=True)

    print(f"[{layer}] all shards pushed, manifest up to date -> {manifest_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--activations-dir", required=True,
        help="perturbed_sweep output dir, e.g. onnx_models/tiny_llama/perturbed_sweep "
        "(layers are discovered from local scratch under this path)",
    )
    ap.add_argument(
        "--layers", nargs="+", default=None,
        help="Only reconstruct these layers (default: every layer with shard_*.h5 on scratch)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rewrite a layer's progress.json/manifest even if one already exists",
    )
    ap.add_argument(
        "--no-push", action="store_true",
        help="Only run pass 1 (write every layer's progress.json/manifest); skip pass 2 (pushing shards) entirely",
    )
    args = ap.parse_args()

    layers = args.layers or _discover_layers(args.activations_dir)
    print(f"{len(layers)} layer(s) to check", flush=True)

    print("=== pass 1/2: writing progress.json + manifest for every layer ===", flush=True)
    to_push = [layer for layer in layers if write_manifest_for_layer(args.activations_dir, layer, force=args.force)]

    if args.no_push:
        print(f"--no-push: skipping pass 2 ({len(to_push)} layer(s) still have unpushed shards)", flush=True)
        return

    print(f"=== pass 2/2: pushing shards for {len(to_push)} layer(s) ===", flush=True)
    for layer in to_push:
        push_layer_shards(args.activations_dir, layer)


if __name__ == "__main__":
    main()
