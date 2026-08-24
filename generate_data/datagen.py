import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def get_token_ids(
    dataset: str,
    tokenizer: str,
    text_column: str = "text",
    split: str = "train",
    dataset_config: str | None = None,
) -> list[list[int]]:
    """Load a HF dataset and tokenize its text column with a HF tokenizer.

    Returns one list of token ids per example (unpadded, no special handling
    of length -- callers assemble these into batches/tensors as needed).
    """
    ds = load_dataset(dataset, dataset_config, split=split)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    return [tok.encode(example[text_column]) for example in ds]


def sample_ambient_points(
    n_points: int,
    dim: int,
    mean: np.ndarray | float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Draw n_points i.i.d. Gaussian points in R^dim with the given mean and std.

    `mean` may be a scalar (broadcast to every coordinate) or a (dim,) vector,
    so callers can place the distribution away from the origin.
    """
    rng = np.random.default_rng(seed)
    mean = np.broadcast_to(mean, (dim,))
    return mean + std * rng.standard_normal((n_points, dim))


def sample_around_point(
    point: np.ndarray,
    n_samples: int,
    min_radius: float = 0.0,
    max_radius: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample n_samples points uniformly in the shell min_radius <= ||x - point|| <= max_radius.

    Direction is uniform on the unit sphere; radius is drawn so that points
    are uniform by volume within the shell (not biased toward the center).
    """
    rng = np.random.default_rng(seed)
    dim = point.shape[-1]

    directions = rng.standard_normal((n_samples, dim))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    u = rng.uniform(0.0, 1.0, size=n_samples)
    radii = (min_radius**dim + u * (max_radius**dim - min_radius**dim)) ** (1.0 / dim)

    return point + directions * radii[:, None]
