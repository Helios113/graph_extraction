import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from config import InputSourceConfig  # add to imports at top
import torch
import tqdm

def sample_ambient_points(
    batch: int,
    seq: int,
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
    return rng.uniform(low=low, high=high, size=(batch, seq, dim))


def sample_around_points_gpu(
    points: torch.Tensor,
    n_samples: int,
    min_radius: float = 0.0,
    max_radius: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = points.device
    batch, seq, dim = points.shape

    directions = torch.randn(
        (n_samples, batch, seq, dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    directions /= directions.norm(dim=-1, keepdim=True)

    u = torch.rand(
        (n_samples, batch, seq, 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    if min_radius == 0.0:
        radii = max_radius * u.pow(1.0 / dim)
    else:
        ratio = min_radius / max_radius
        radii = max_radius * (ratio**dim + u * (1.0 - ratio**dim)).pow(1.0 / dim)

    directions.mul_(radii)
    return directions.add_(points.unsqueeze(0))


def sample_around_points(
    points: np.ndarray,
    n_samples: int,
    min_radius: float = 0.0,
    max_radius: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    batch, seq, dim = points.shape

    directions = rng.standard_normal((n_samples, batch, seq, dim), dtype=np.float32)
    norm_sq = np.einsum("nbsd,nbsd->nbs", directions, directions)[..., None]
    norm = np.sqrt(norm_sq)

    u = rng.uniform(0.0, 1.0, size=(n_samples, batch, seq, 1))
    if min_radius == 0.0:
        radii = max_radius * u ** (1.0 / dim)
    else:
        ratio = min_radius / max_radius
        radii = max_radius * (ratio**dim + u * (1.0 - ratio**dim)) ** (1.0 / dim)

    scale = (radii / norm).astype(np.float32)
    directions *= scale  # combines normalize + radius-scale into one pass
    np.add(directions, points[None, ...], out=directions)  # in-place add
    perturbed = directions

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

    ds = load_dataset(
        input_source.dataset, input_source.dataset_config, split=input_source.split
    )
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
        raise ValueError(
            f"dataset only yielded {len(rows)} examples with >= {seq_len} tokens, need {batch}"
        )

    return np.asarray(rows, dtype=np.int64)


def knn_gpu(
    points: torch.Tensor,
    k: int = 5,
    query_block: int = 4096,
    ref_block: int = 16384,
    return_distances: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Computes brute-force exact k-NN using blockwise squared Euclidean distance
    and a running top-k merge.

    Args:
        points: (N, D) float32 tensor on GPU.
        k: Number of nearest neighbors to find.
        query_block: Chunk size for query points.
        ref_block: Chunk size for reference points.
        return_distances: If True, also materialize the (N, k) distances
            tensor on GPU. Off by default: most callers only need the
            indices (e.g. to gather neighbor rows for a downstream
            computation), and at N in the millions the (N, k) distances
            tensor is itself tens of GB, so skipping it when unused matters.

    Returns:
        distances: (N, k) tensor of Euclidean distances, sorted ascending,
            on `points.device` -- or None if return_distances is False.
        indices: (N, k) int32 tensor of neighbor indices, in pinned CPU
            memory (not on `points.device`) -- kept off-GPU and in int32
            (not int64) since at N in the millions the full (N, k) index
            matrix is tens of GB and doesn't need to be GPU-resident or
            8-byte-per-entry. Callers that need a block of it on-device
            should slice the CPU tensor and move just that block over, e.g.
            `indices[start:end].to(device)`.
    """
    N, _ = points.shape
    device = points.device
    dtype = points.dtype

    # Precompute squared norms for the distance formula: ||q||^2 + ||r||^2 - 2(q @ r.T)
    sq_norms = (points ** 2).sum(dim=1)

    out_dist = torch.empty((N, k), dtype=dtype, device=device) if return_distances else None
    # int32 (not int64): out_idx is (N, k) and can be tens of GB in pinned
    # host memory at N/k in the millions -- int32 halves that, and N fits
    # comfortably under int32's ~2.1B range for any sweep size this repo
    # produces.
    assert N < 2**31, f"N={N} too large for int32 neighbor indices"
    out_idx = torch.empty((N, k), dtype=torch.int32, pin_memory=True)

    for q_start in range(0, N, query_block):
        q_end = min(q_start + query_block, N)
        q = points[q_start:q_end]
        q_sq = sq_norms[q_start:q_end].unsqueeze(1)

        best_dist, best_idx = None, None

        for ref_start in range(0, N, ref_block):
            ref_end = min(ref_start + ref_block, N)
            ref = points[ref_start:ref_end]
            ref_sq = sq_norms[ref_start:ref_end].unsqueeze(0)

            # 1. Compute pairwise squared distances for current blocks
            d2 = q_sq + ref_sq - 2.0 * (q @ ref.T)
            d2.clamp_min_(0.0)

            # 2. Find local top-k for the reference block
            chunk_k = min(k, ref_end - ref_start)
            chunk_dist, chunk_local_idx = torch.topk(
                d2, chunk_k, dim=1, largest=False, sorted=True
            )
            chunk_global_idx = chunk_local_idx + ref_start

            # 3. Merge with running best results
            if best_dist is None:
                best_dist = chunk_dist
                best_idx = chunk_global_idx
            else:
                merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
                merged_idx = torch.cat([best_idx, chunk_global_idx], dim=1)

                take_k = min(k, merged_dist.shape[1])
                best_dist, sel = torch.topk(
                    merged_dist, take_k, dim=1, largest=False, sorted=True
                )
                best_idx = torch.gather(merged_idx, 1, sel)

        # Convert squared distances to Euclidean distances
        if out_dist is not None:
            out_dist[q_start:q_end] = best_dist.sqrt()
        out_idx[q_start:q_end] = best_idx.to(torch.int32).to("cpu", non_blocking=True)

    return out_dist, out_idx




def estimate_local_tangent_space_p_var(
    neighbors: torch.Tensor, 
    p: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    neighbors: (K, D) tensor, where K is elements and D is d_model.
    p: Target variance fraction to explain (0.0 to 1.0).
    
    Returns:
        tangent_basis: (k, D) tensor. Only the required basis vectors are returned.
        k: Scalar integer tensor indicating the exact number of components used.
    """
    K, D = neighbors.shape
    
    # 1. Center the neighborhood points
    local_mean = neighbors.mean(dim=0, keepdim=True)
    centered = neighbors - local_mean

    # 2. Perform SVD
    U, S, Vh = torch.linalg.svd(centered.to(torch.float32), full_matrices=False)
    if p == 1:
        return Vh, Vh.shape[0]
    # 3. Calculate explained variance ratio
    eigenvalues = S ** 2
    total_variance = eigenvalues.sum(dim=-1, keepdim=True)
    explained_variance_ratio = eigenvalues.cumsum(dim=-1) / total_variance

    # 4. Determine number of components needed to reach p
    mask = explained_variance_ratio < p
    
    # The required index is the number of True values + 1
    # Max intrinsic dim is K-1 because centering removes 1 degree of freedom
    k = mask.sum() + 1
    k = torch.clamp(k, max=K-1)
    
    # 5. Extract the basis
    tangent_basis = Vh[:k, :]

    return tangent_basis, k


def sample_ball_in_subspace(basis, center, radius, n_samples, rng=None):
    """
    Sample points uniformly (by volume) from a d-dimensional ball
    embedded in R^n, lying in the subspace spanned by `basis`,
    centered at `center`.

    Parameters
    ----------
    basis : (n, d) array
        Columns are basis vectors spanning the subspace (need not be orthonormal).
    center : (n,) array
        Center of the ball in ambient space.
    radius : float
        Ball radius.
    n_samples : int
    rng : np.random.Generator, optional

    Returns
    -------
    (n_samples, n) array of points in ambient space.
    """
    if rng is None:
        rng = np.random.default_rng()

    basis = np.asarray(basis, dtype=float)
    n, d = basis.shape

    # Orthonormal basis for the subspace
    Q, _ = np.linalg.qr(basis)  # (n, d), orthonormal columns

    # Uniform sampling in d-ball:
    # 1. direction: normalize a Gaussian vector -> uniform on S^{d-1}
    # 2. radius: r = R * u^{1/d}, u ~ Uniform(0,1), for correct volume density
    g = rng.normal(size=(n_samples, d))
    directions = g / np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.uniform(size=(n_samples, 1))
    radii = radius * u ** (1.0 / d)
    local_points = directions * radii  # (n_samples, d), coords in subspace basis Q

    # Map to ambient space
    points = center + local_points @ Q.T  # (n_samples, n)
    return points