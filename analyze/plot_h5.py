"""Load an .h5 file and plot datasets from it.

Usage:
    uv run analyze/plot_h5.py path/to/file.h5
    uv run analyze/plot_h5.py path/to/file.h5 --dataset jacobians
"""

import argparse




def print_structure(h5file: h5py.File) -> None:
    print(f"Contents of {h5file.filename}:")

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")

    h5file.visititems(visitor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Path to the .h5 file to load.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Name of the dataset to plot. Defaults to the first dataset found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.h5_path, "r") as f:
        print_structure(f)

        dataset_name = args.dataset
        if dataset_name is None:
            dataset_name = next(iter(f.keys()))
            print(f"\nNo --dataset given, defaulting to '{dataset_name}'")

        data = np.asarray(f[dataset_name])

    print(f"\nLoaded '{dataset_name}' with shape {data.shape}, dtype {data.dtype}")

    # --- Matplotlib boilerplate: edit below to plot what you're interested in ---
    fig, ax = plt.subplots(figsize=(8, 5))

    if data.ndim == 1:
        ax.plot(data)
    elif data.ndim == 2:
        im = ax.imshow(data, aspect="auto")
        fig.colorbar(im, ax=ax)
    else:
        flat = data.reshape(data.shape[0], -1)
        ax.plot(flat)

    ax.set_title(dataset_name)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
