import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from config import InputSourceConfig  # add to imports at top



def sample_ambient_points(
    batch: int,
    seq:int,
    dim: int,
    low: np.ndarray | float = 0.0,
    high: np.ndarray | float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Draw n_points i.i.d.

    points uniformly distributed in [low, high)^dim.

    `low` and `high` can each be a scalar or a (dim,) vector.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(batch,seq, dim))

def sample_around_points(
    points: np.ndarray,
    n_samples: int,
    min_radius: float = 0.0,
    max_radius: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    batch, seq, dim = points.shape

    directions = rng.standard_normal((n_samples, batch, seq, dim))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    u = rng.uniform(0.0, 1.0, size=(n_samples, batch, seq, 1))
    radii = (min_radius**dim + u * (max_radius**dim - min_radius**dim)) ** (
        1.0 / dim
    )

    perturbed = points[None, ...] + directions * radii
    print(perturbed.shape)

    return perturbed

def repeat_points_to_match(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Duplicate input points along a new sample dimension to match the shape

    of the perturbed points output by `sample_around_points`.

    Parameters:
    -----------
    points : np.ndarray
        Array of shape (dim,) or (n_points, dim).
    n_samples : int
        Number of repetitions per point.

    Returns:
    --------
    np.ndarray
        - Shape (n_samples, dim) if `points` was 1D (dim,).
        - Shape (n_points, n_samples, dim) if `points` was 2D (n_points, dim).
    """
    points_arr = np.asarray(points)
    is_1d = points_arr.ndim == 1

    if is_1d:
        return np.repeat(points_arr[None, :], repeats=n_samples, axis=0)

    # For 2D inputs, expand to (n_points, 1, dim) then broadcast/repeat along axis 1
    return np.repeat(points_arr[:, None, :], repeats=n_samples, axis=1)

def get_token_ids_array(
    input_source: InputSourceConfig,
    seq_len: int,
) -> np.ndarray:

    ds = load_dataset(input_source.dataset, input_source.dataset_config, split=input_source.split)
    tok = AutoTokenizer.from_pretrained(input_source.tokenizer)

    rows: list[list[int]] = []
    for example in ds:
        if len(rows) >= input_source.data_batch_size:
            break
        encoded = tok.encode(example[input_source.text_column])
        if len(encoded) < seq_len:
            continue
        rows.append(encoded[:seq_len])

    if len(rows) < input_source.data_batch_size:
        raise ValueError(f"dataset only yielded {len(rows)} examples with >= {seq_len} tokens, need {batch}")

    return np.asarray(rows, dtype=np.int64)
