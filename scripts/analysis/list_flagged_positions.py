"""Lists (sample, pos) pairs from a lipschitz_ratio_bootstrap_eqd_sweep_gpu.py
(or CPU lipschitz_ratio_eqd_fit.py) output directory whose fit looks
suspect: xi >= 0 (GPD tail doesn't have a finite lower bound, so c_hat is
undefined) or c_hat < 0 (an estimated co-Lipschitz infimum below zero, which
isn't physically valid for a ratio of norms).

Prints one "<sample_idx> <pos>" pair per line, so it composes with xargs /
a shell loop to drive scripts/python/plot_lipschitz_ratio_eqd_fit.py's
--mode detail over just the flagged positions.

Usage:
    uv run python scripts/analysis/list_flagged_positions.py \\
        onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4
"""

import argparse
import glob
import os

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair_dir", type=str)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pair_dir, "sample_*.h5")))
    if not paths:
        raise SystemExit(f"no sample_*.h5 files found under {args.pair_dir!r}")

    for path in paths:
        sample_idx = int(os.path.basename(path).removeprefix("sample_").removesuffix(".h5"))
        with h5py.File(path, "r") as h5f:
            for pos_key in sorted(k for k in h5f.keys() if k.startswith("pos_")):
                grp = h5f[pos_key]
                if "fit_error" in grp.attrs:
                    continue
                xi = grp.attrs["shape_xi"]
                c_hat = grp.attrs["c_hat"]
                if xi >= 0 or c_hat < 0:
                    print(f"{sample_idx} {grp.attrs['pos']}")


if __name__ == "__main__":
    main()
