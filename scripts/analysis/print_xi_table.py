"""Prints a (sample, pos) -> shape_xi table for one sublayer-pair directory
of lipschitz_ratio_bootstrap_eqd_sweep_gpu.py output.

Usage:
    uv run python scripts/analysis/print_xi_table.py \\
        onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4
"""

import argparse
import glob
import os

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_dir", type=str, help="e.g. onnx_models/.../embedding.add_4")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pair_dir, "sample_*.h5")))
    if not paths:
        raise SystemExit(f"no sample_*.h5 files found under {args.pair_dir!r}")

    print(f"{'sample':>8} {'pos':>6} {'xi':>12}")
    for path in paths:
        sample_idx = int(os.path.basename(path).removeprefix("sample_").removesuffix(".h5"))
        with h5py.File(path, "r") as h5f:
            for pos_key in sorted(k for k in h5f.keys() if k.startswith("pos_")):
                grp = h5f[pos_key]
                pos = grp.attrs["pos"]
                if "fit_error" in grp.attrs:
                    print(f"{sample_idx:>8} {pos:>6} {'ERROR':>12}")
                    continue
                xi = grp.attrs["shape_xi"]
                print(f"{sample_idx:>8} {pos:>6} {xi:>12.6f}")


if __name__ == "__main__":
    main()
