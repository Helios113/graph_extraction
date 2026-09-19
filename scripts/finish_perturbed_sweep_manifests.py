"""Finishes layers whose perturbed-sweep compute is fully done (progress.json
shows next_row >= n_rows) but whose manifest was never written -- e.g. a
sweep job that hit its SLURM --time limit and was TIMEOUT-killed before
reaching writer.close() for any layer. Since every layer in a sweep run
advances through the same chunk loop in lockstep (see
scripts.generate_perturbed_sweep._run_perturbed_sweep_once), a timeout partway
through the run typically leaves every layer's data collection complete on
local scratch, just not pushed/manifested -- close() is the only thing that
never ran.

This does NOT recompute anything. It only:
  1. Pushes each layer's not-yet-pushed shards (from local scratch, per
     progress.json's completed_shards), ONE AT A TIME -- no concurrent
     uploads, unlike the sweep's own AsyncPusher -- so this can run
     unattended without competing with other jobs' SFTP connections.
  2. Writes that layer's manifest once every shard is confirmed pushed.

Run on the node that actually holds the local scratch data (e.g. via `srun
--partition=interactive --nodelist=mauao`), not the NFS login node -- shards
not yet pushed only exist on that node's /local/scratch.
"""

import argparse
import glob
import json
import os
import time

from execute.remote_storage import push
from execute.sharded_writer import _scratch_root, write_manifest


def _progress_paths(activations_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(activations_dir, "*.progress.json")))


def finish_layer(progress_path: str, force: bool) -> None:
    layer_key = progress_path[: -len(".progress.json")]
    manifest_path = layer_key + ".manifest.json"
    layer = os.path.basename(layer_key)

    with open(progress_path) as f:
        progress = json.load(f)

    if progress["next_row"] < progress["n_rows"]:
        print(
            f"[{layer}] skipping: compute incomplete "
            f"({progress['next_row']}/{progress['n_rows']} rows) -- "
            "this layer needs its sweep re-run, not just a manifest fix",
            flush=True,
        )
        return

    if os.path.exists(manifest_path) and not force:
        print(f"[{layer}] skipping: manifest already exists (pass --force to rewrite)", flush=True)
        return

    unpushed = [s for s in progress["completed_shards"] if not s["pushed"]]
    print(
        f"[{layer}] {len(progress['completed_shards'])} shard(s), "
        f"{len(unpushed)} not yet pushed",
        flush=True,
    )

    for shard in unpushed:
        scratch_path = os.path.join(
            _scratch_root(), layer_key, f"shard_{shard['shard_idx']:05d}.h5",
        )
        if not os.path.exists(scratch_path):
            print(
                f"[{layer}] shard {shard['shard_idx']}: no local file at "
                f"{scratch_path} -- can't push what isn't there, skipping "
                "this layer's manifest",
                flush=True,
            )
            return

        t0 = time.time()
        print(f"[{layer}] pushing shard {shard['shard_idx']} ({scratch_path})...", flush=True)
        push(scratch_path, virtual_path=shard["virtual_path"])
        shard["pushed"] = True
        progress["updated_at"] = time.time()
        tmp_path = progress_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp_path, progress_path)
        print(f"[{layer}] shard {shard['shard_idx']} pushed in {time.time() - t0:.1f}s", flush=True)

    write_manifest(".", layer_key, progress)
    print(f"[{layer}] manifest written -> {manifest_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--activations-dir", required=True,
        help="perturbed_sweep output dir (contains one *.progress.json per layer)",
    )
    ap.add_argument(
        "--layers", nargs="+", default=None,
        help="Only finish these layers (default: every layer with a progress.json)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rewrite a layer's manifest even if one already exists",
    )
    args = ap.parse_args()

    progress_paths = _progress_paths(args.activations_dir)
    if args.layers is not None:
        wanted = set(args.layers)
        progress_paths = [
            p for p in progress_paths
            if os.path.basename(p)[: -len(".progress.json")] in wanted
        ]
    print(f"{len(progress_paths)} layer(s) to check", flush=True)

    for progress_path in progress_paths:
        finish_layer(progress_path, force=args.force)


if __name__ == "__main__":
    main()
