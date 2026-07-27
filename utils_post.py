import time

from datasets import load_dataset
from transformers import AutoTokenizer

import h5py
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from config import ModelConfig, grad_names
from optimise_activations import sample_points

def _session_options() -> ort.SessionOptions:
    """Explicit thread counts, since letting ORT pick its own can crash on machines where
    pthread_setaffinity_np rejects its auto-computed affinity mask."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def _save_activations(f: h5py.File, activations: dict[str, np.ndarray]) -> None:
    group = f.create_group("activations")
    for name, array in activations.items():
        # Some union-graph outputs (e.g. the ReduceSum'd loss) are 0-d scalars: h5py
        # rejects chunk/filter options (like gzip) on scalar datasets.
        group.create_dataset(name, data=array, compression="gzip" if array.ndim > 0 else None)


def save_diagonal_jacobians_svd(cfg: ModelConfig, diagonal_blocks: dict[str, np.ndarray], activations: dict[str, np.ndarray]) -> None:
    """SVD (per leading-dim row -- SEQ position, or (sample, position) if cfg.samples is set)
    of each pair's diagonal blocks, saved as three gzip-compressed stacked datasets U/S/Vh
    (each stacked over pairs, ordered by cfg.pairs), with a `pairs` attr.
    Also saves the union graph's forward-pass `activations` under an `activations/` group.
    """
    tags = [f"{upstream}_{downstream}" for upstream, downstream in cfg.pairs]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Us, Ss, Vhs = [], [], []
    for upstream, _ in cfg.pairs:
        blocks = torch.from_numpy(diagonal_blocks[upstream]).to(device)  # (SEQ, D_MODEL, D_MODEL) or (samples, num_positions, D_MODEL, D_MODEL)
        svd_start = time.perf_counter()
        U, S, Vh = torch.linalg.svd(blocks)  # batched SVD, one gesdd/gesvdj call per leading-dim row
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  SVD of {upstream} ({device}): {time.perf_counter() - svd_start:.3f}s")
        Us.append(U.cpu().numpy())
        Ss.append(S.cpu().numpy())
        Vhs.append(Vh.cpu().numpy())

    cfg.svd_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.svd_path, "w") as f:
        f.create_dataset("U", data=np.stack(Us), compression="gzip")
        f.create_dataset("S", data=np.stack(Ss), compression="gzip")
        f.create_dataset("Vh", data=np.stack(Vhs), compression="gzip")
        f.attrs["pairs"] = tags
        _save_activations(f, activations)
    print(f"Saved diagonal-block SVD -> {cfg.svd_path}")


def save_diagonal_jacobians(cfg: ModelConfig, diagonal_blocks: dict[str, np.ndarray], activations: dict[str, np.ndarray]) -> None:
    """Save {upstream: (SEQ, D_MODEL, D_MODEL)} (no sampling) or {upstream: (samples,
    num_positions, D_MODEL, D_MODEL)} (cfg.samples set) as one gzip-compressed stacked
    dataset, ordered by cfg.pairs, with a `pairs` attr recording each row's
    "{upstream}_{downstream}" tag. Also saves each block's sign/log|det| (np.linalg.slogdet,
    batched over the same leading axes as diagonal_blocks, prefixed with the pairs axis, so
    `sign`/`logabsdet`'s leading-index entry is that pair/position(-and-sample)'s determinant
    sign and log-abs-determinant) and the union graph's forward-pass `activations` under an
    `activations/` group.
    """
    tags = [f"{upstream}_{downstream}" for upstream, downstream in cfg.pairs]
    stacked = np.stack([diagonal_blocks[upstream] for upstream, _ in cfg.pairs])  # (num_pairs, SEQ, D_MODEL, D_MODEL) or (num_pairs, samples, num_positions, D_MODEL, D_MODEL)
    sign, logabsdet = np.linalg.slogdet(stacked)  # same leading axes as stacked, minus the last two

    cfg.diagonal_jacobians_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.diagonal_jacobians_path, "w") as f:
        f.create_dataset("diagonal_blocks", data=stacked, compression="gzip")
        f.create_dataset("sign", data=sign, compression="gzip")
        f.create_dataset("logabsdet", data=logabsdet, compression="gzip")
        f.attrs["pairs"] = tags
        _save_activations(f, activations)
    print(f"Saved diagonal Jacobians {stacked.shape} -> {cfg.diagonal_jacobians_path}")


def _tokenized_samples(cfg: ModelConfig, num_examples: int) -> np.ndarray:
    """num_examples consecutive, non-overlapping `seq`-token windows from
    `input_source.dataset` tokenized with `input_source.tokenizer`, shape (num_examples, seq).
    Only valid for cfg.input_name == "input_ids".
    """
    src = cfg.input_source
    assert cfg.input_name == "input_ids", f"{cfg.name}: input_source.mode='dataset' requires input_name='input_ids'"
    tokenizer = AutoTokenizer.from_pretrained(src.tokenizer)
    ds = load_dataset(src.dataset, src.dataset_config, split=src.split, streaming=True)

    needed = cfg.seq * num_examples
    token_ids: list[int] = []
    for example in ds:
        token_ids.extend(tokenizer(example[src.text_column])["input_ids"])
        if len(token_ids) >= needed:
            break
    if len(token_ids) < needed:
        raise RuntimeError(
            f"{cfg.name}: dataset {src.dataset!r} yielded only {len(token_ids)} tokens, need {needed}"
        )

    return np.array(token_ids[:needed], dtype=np.int64).reshape(num_examples, cfg.seq)


def _tokenized_sample(cfg: ModelConfig) -> np.ndarray:
    """First `seq`-token window from `input_source.dataset`, shape (1, seq)."""
    return _tokenized_samples(cfg, 1)


def sample_jacobian_points(cfg: ModelConfig, sublayer_idx, position) -> np.ndarray:
    """Single (1, seq[, input_dim]) example used as the shared batch content for
    diagonal-Jacobian computation -- random per input_source.mode='random' (default), or a
    real tokenized window per input_source.mode='dataset'.
    """
    if cfg.input_source.mode == "fixed_norm":
        return sample_points(cfg, sublayer_idx=sublayer_idx, position=position, seed=0)
    if cfg.input_source.mode == "dataset":
        return _tokenized_sample(cfg)
    np.random.seed(0)
    if cfg.input_name == "input_ids":
        return np.random.randint(0, cfg.vocab_size, (1, cfg.seq)).astype(np.int64)
    assert cfg.input_dim is not None, f"{cfg.name}: input_dim required in config for non-input_ids models"
    if cfg.input_name == "random":
        return np.random.randn(1, cfg.seq, cfg.input_dim).astype(np.float32)
    raise ValueError


def _sample_examples_np(cfg: ModelConfig, rng: np.random.Generator, num_examples: int) -> np.ndarray:
    """num_examples independent (num_examples, seq[, input_dim])-shaped inputs -- consecutive
    tokenized dataset windows per input_source.mode='dataset', else independent random draws."""
    if cfg.input_source.mode == "dataset":
        return _tokenized_samples(cfg, num_examples)
    if cfg.input_name == "input_ids":
        return rng.integers(0, cfg.vocab_size, (num_examples, cfg.seq)).astype(np.int64)
    assert cfg.input_dim is not None, f"{cfg.name}: input_dim required in config for non-input_ids models"
    return rng.standard_normal((num_examples, cfg.seq, cfg.input_dim)).astype(np.float32)


def sample_diagonal_jacobians(
    cfg: ModelConfig,
    session: ort.InferenceSession | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calls compute_diagonal_jacobians once per independent input (cfg.samples of them --
    independent random draws, or consecutive dataset windows per input_source.mode='dataset'),
    reusing one session, and stacks the results: {upstream_name: (samples, num_positions,
    D_MODEL, D_MODEL)}, where num_positions is cfg.seq (cfg.jacobian_position is None) or 1
    (cfg.jacobian_position is an int). Each call keeps compute_diagonal_jacobians's own
    cfg.batch-chunking over its mask rows (num_positions * d_model), so cfg.batch just needs
    to divide that -- same requirement as the non-sampling path, independent of cfg.samples.

    Activations are returned from the first sample only.
    """
    assert cfg.samples is not None
    union_model_path = cfg.gradient_path
    if session is None:
        session = ort.InferenceSession(union_model_path, sess_options=_session_options(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    rng = np.random.default_rng(cfg.seed)
    # Drawn upfront: dataset mode streams cfg.samples consecutive, non-overlapping windows,
    # so each call must slice into one shared draw rather than re-reading the stream from
    # its start every call.
    all_examples_np = _sample_examples_np(cfg, rng, cfg.samples)  # (samples, seq[, input_dim])

    per_sample_blocks = []
    activations: dict[str, np.ndarray] = {}
    for i in tqdm(range(cfg.samples), desc="sampling Jacobians"):
        example_np = all_examples_np[i:i + 1]  # (1, seq[, input_dim])
        blocks, sample_activations = compute_diagonal_jacobians(
            cfg, example_input_np=example_np, session=session, disable_progress=True,
        )
        per_sample_blocks.append(blocks)  # {upstream: (num_positions, D_MODEL, D_MODEL)}
        if i == 0:
            activations = sample_activations

    all_blocks = {
        upstream: np.stack([blocks[upstream] for blocks in per_sample_blocks])
        for upstream, _ in cfg.pairs
    }  # {upstream: (samples, num_positions, D_MODEL, D_MODEL)}
    return all_blocks, activations

def inv_sqrt_AtA_eig(A):
    """(AtA)^-1/2 batched over A's leading dims, where A is (..., D_MODEL, D_MODEL) --
    A.T (transposing only the last two axes, not every axis like the bare `.T` does on an
    N-d array) and a broadcasted eigvals**-0.5 scaling (not np.diag, which only accepts
    1D/2D input) so this works on stacked_fp64's full (num_pairs, num_positions, D_MODEL,
    D_MODEL) shape in one call."""
    AtA = np.swapaxes(A, -1, -2) @ A
    eigvals, eigvecs = np.linalg.eigh(AtA)
    return (eigvecs * eigvals[..., None, :] ** -0.5) @ np.swapaxes(eigvecs, -1, -2)


def sample_diagonal_jacobians_slogdet(
    cfg: ModelConfig,
    session: ort.InferenceSession | None = None,
) -> None:
    """Like sample_diagonal_jacobians, but never accumulates samples' full (D_MODEL, D_MODEL)
    blocks: each sample's block is slogdet'd immediately after that sample's pass and appended
    to cfg.diagonal_jacobians_path's `sign`/`logabsdet`/`cond`/`smallest_sv`/`largest_sv`
    datasets (shape (num_pairs, samples, num_positions), stacked/ordered by cfg.pairs like
    save_diagonal_jacobians), then discarded. `cond` is each block's condition number
    (largest/smallest singular value, computed in fp64), recorded to check whether negative
    slogdets coincide with ill-conditioned blocks. Peak memory no longer grows with
    cfg.samples, and the h5 file is flushed after every sample, so a crash mid-run leaves
    every sample computed so far already on disk (readable even from a partial file -- h5py
    reports however many rows were actually flushed).

    No `diagonal_blocks` dataset is saved (the raw blocks are never kept around to save), and
    there is no SVD variant of this function -- an SVD needs the raw block, which is exactly
    what this trades away for the constant-memory/crash-safety property.
    """
    assert cfg.samples is not None
    union_model_path = cfg.gradient_path
    if session is None:
        session = ort.InferenceSession(union_model_path, sess_options=_session_options(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    rng = np.random.default_rng(cfg.seed)
    all_examples_np = _sample_examples_np(cfg, rng, cfg.samples)  # (samples, seq[, input_dim])
    tags = [f"{upstream}_{downstream}" for upstream, downstream in cfg.pairs]
    num_pairs = len(cfg.pairs)

    cfg.diagonal_jacobians_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.diagonal_jacobians_path, "w") as f:
        sign_dset = logabsdet_dset = cond_dset = smallest_sv_dset = largest_sv_dset = None
        f.attrs["pairs"] = tags
        f.attrs["samples_written"] = 0
        for i in tqdm(range(cfg.samples), desc="sampling Jacobians (slogdet only)"):
            example_np = all_examples_np[i:i + 1]  # (1, seq[, input_dim])
            blocks, activations = compute_diagonal_jacobians(
                cfg, example_input_np=example_np, session=session, disable_progress=True,
            )  # blocks: {upstream: (num_positions, D_MODEL, D_MODEL)}
            stacked = np.stack([blocks[upstream] for upstream, _ in cfg.pairs])  # (num_pairs, num_positions, D_MODEL, D_MODEL)
            stacked_fp64 = stacked.astype(np.float64)
            
            
            M = inv_sqrt_AtA_eig(stacked_fp64)
            stacked_fp64 = stacked_fp64 @ M
            
            svals = np.linalg.svd(stacked_fp64, compute_uv=False)  # (num_pairs, num_positions, D_MODEL), descending
            smallest_sv = svals[..., -1]
            largest_sv = svals[..., 0]
            
            cond = largest_sv / smallest_sv  # (num_pairs, num_positions)
            sign, logabsdet = np.linalg.slogdet(stacked_fp64)
          
          
            # each (num_pairs, num_positions)
            if sign_dset is None:
                num_positions = sign.shape[1]
                sign_dset = f.create_dataset(
                    "sign", shape=(num_pairs, 0, num_positions),
                    maxshape=(num_pairs, None, num_positions), dtype=sign.dtype, compression="gzip",
                )
                logabsdet_dset = f.create_dataset(
                    "logabsdet", shape=(num_pairs, 0, num_positions),
                    maxshape=(num_pairs, None, num_positions), dtype=logabsdet.dtype, compression="gzip",
                )
                cond_dset = f.create_dataset(
                    "cond", shape=(num_pairs, 0, num_positions),
                    maxshape=(num_pairs, None, num_positions), dtype=cond.dtype, compression="gzip",
                )
                smallest_sv_dset = f.create_dataset(
                    "smallest_sv", shape=(num_pairs, 0, num_positions),
                    maxshape=(num_pairs, None, num_positions), dtype=smallest_sv.dtype, compression="gzip",
                )
                largest_sv_dset = f.create_dataset(
                    "largest_sv", shape=(num_pairs, 0, num_positions),
                    maxshape=(num_pairs, None, num_positions), dtype=largest_sv.dtype, compression="gzip",
                )
                if i == 0:
                    _save_activations(f, activations)

            sign_dset.resize(i + 1, axis=1)
            logabsdet_dset.resize(i + 1, axis=1)
            cond_dset.resize(i + 1, axis=1)
            smallest_sv_dset.resize(i + 1, axis=1)
            largest_sv_dset.resize(i + 1, axis=1)
            sign_dset[:, i, :] = sign
            logabsdet_dset[:, i, :] = logabsdet
            cond_dset[:, i, :] = cond
            smallest_sv_dset[:, i, :] = smallest_sv
            largest_sv_dset[:, i, :] = largest_sv
            f.attrs["samples_written"] = i + 1
            f.flush()

    print(f"Saved {cfg.samples} samples' sign/logabsdet/cond/smallest_sv/largest_sv -> {cfg.diagonal_jacobians_path}")


def compute_diagonal_jacobians(
    cfg: ModelConfig,
    example_input_np: np.ndarray | None = None,
    session: ort.InferenceSession | None = None,
    disable_progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    
    positions = range(cfg.seq) if cfg.jacobian_position is None else [cfg.jacobian_position % cfg.seq]
    num_elements = len(positions) * cfg.d_model
    union_model_path = cfg.gradient_path

    # convert_model.py's _add_grad_outputs already appended each pair's grad as a graph
    # output, and Cast-to-float32'd every bf16 output (grads and pre-existing forward-pass
    # activations alike) -- ORT's Python bindings can never convert a bf16 tensor to numpy
    # (session.run and OrtValue.numpy() both raise "No corresponding Numpy type for Tensor
    # Type"; there's no io_binding workaround, it's baked into the compiled extension), so
    # the *_f32 cast output is what gets fetched here, not the raw tensor.
    if session is None:
        session = ort.InferenceSession(union_model_path, sess_options=_session_options(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    session_output_names = {o.name for o in session.get_outputs()}

    def _f32_output_name(tensor_name: str) -> str:
        cast_name = f"{tensor_name}_f32"
        return cast_name if cast_name in session_output_names else tensor_name

    raw_grad_names = grad_names(cfg)
    grad_output_names = [_f32_output_name(g) for g in raw_grad_names]
    grad_related_names = set(raw_grad_names) | set(grad_output_names)
    # Every bf16 forward output has both a raw name (unfetchable -- ORT can't convert bf16
    # to numpy) and a `{name}_f32` cast counterpart in session_output_names; skip the raw
    # name whenever its cast counterpart also exists; float32 outputs have no such
    # counterpart, so `{name}_f32` just doesn't appear in session_output_names for those.
    forward_output_names = [
        name for name in session_output_names
        if name not in grad_related_names and f"{name}_f32" not in session_output_names
    ]
    grad_np_dtype = np.float32

    mask_full_np = np.zeros((num_elements, cfg.seq, cfg.d_model), dtype=np.float32)
    for row, position in enumerate(positions):
        mask_full_np[row * cfg.d_model:(row + 1) * cfg.d_model, position, :] = np.eye(cfg.d_model, dtype=np.float32)
    example_np = example_input_np if example_input_np is not None else sample_points(cfg)
    input_batch_np = np.tile(example_np, (cfg.batch,) + (1,) * (example_np.ndim - 1))

    if num_elements % cfg.batch != 0:
        raise ValueError(
            f"{cfg.name}: batch ({cfg.batch}) must evenly divide {num_elements} "
            f"({'seq*d_model' if cfg.jacobian_position is None else 'd_model'}) -- "
            "the ORT session's batch dim is fixed at export time, so a remainder chunk can't be run."
        )
    num_chunks = num_elements // cfg.batch
    full_jacobians = {upstream: np.zeros((num_elements, cfg.seq, cfg.d_model), dtype=grad_np_dtype) for upstream, _ in cfg.pairs}
    activations: dict[str, np.ndarray] = {}

    for chunk_idx in tqdm(range(num_chunks), desc="ORT union graph", disable=disable_progress):
        chunk_mask = mask_full_np[chunk_idx * cfg.batch:(chunk_idx + 1) * cfg.batch]
        # Forward-pass activations don't depend on chunk_mask (only the backprop seed does),
        # so they're identical across chunks -- fetch them once, on the first chunk.
        fetch_names = grad_output_names + (forward_output_names if chunk_idx == 0 else [])
        outputs = session.run(
            fetch_names,
            {cfg.input_name: input_batch_np, "mask": chunk_mask},
        )
        out = dict(zip(fetch_names, outputs))
        for (upstream, _), out_name in zip(cfg.pairs, grad_output_names):
            full_jacobians[upstream][chunk_idx * cfg.batch:(chunk_idx + 1) * cfg.batch] = out[out_name]
        if chunk_idx == 0:
            activations = {name: out[name] for name in forward_output_names}

    diagonal_blocks = {}
    for upstream, _ in cfg.pairs:
        # full_jacobians[upstream] is (num_elements, seq, d_model): row block r (d_model
        # rows) holds d(downstream)/d(upstream) for output position positions[r], so
        # [r*d_model:(r+1)*d_model, positions[r], :] is that position's diagonal block.
        # Cast to fp32 regardless of the model's dtype: downstream consumers (h5py,
        # torch.linalg.svd) don't support ml_dtypes' bfloat16.
        jacobian = full_jacobians[upstream]
        diagonal_blocks[upstream] = np.stack([
            jacobian[row * cfg.d_model:(row + 1) * cfg.d_model, position, :]
            for row, position in enumerate(positions)
        ]).astype(np.float32)
    return diagonal_blocks, activations


def save_sublayer_jacobian(
    cfg: ModelConfig, sublayer_idx: int, diagonal_block: np.ndarray, activations: dict[str, np.ndarray],
) -> None:
    """Saves one sublayer's diagonal Jacobian block(s) (SEQ, D_MODEL, D_MODEL) plus its
    sign/log|det| (np.linalg.slogdet, batched over the SEQ axis) and forward-pass
    `activations`, to cfg.sublayer_jacobians_dir/sublayer{idx}_{input}_{output}.h5.
    """
    sublayer = cfg.sublayers[sublayer_idx]
    tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
    sign, logabsdet = np.linalg.slogdet(diagonal_block)

    out_path = cfg.sublayer_jacobians_dir / f"{tag}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("diagonal_block", data=diagonal_block, compression="gzip")
        f.create_dataset("sign", data=sign, compression="gzip")
        f.create_dataset("logabsdet", data=logabsdet, compression="gzip")
        f.attrs["pair"] = f"{sublayer.input}_{sublayer.output}"
        _save_activations(f, activations)
    print(f"Saved sublayer Jacobian {diagonal_block.shape} -> {out_path}")


def save_collision_opt_results(cfg: ModelConfig, all_results: dict[int, dict[str, np.ndarray]]) -> None:
    cfg.collision_opt_results_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.collision_opt_results_path, "w") as f:
        for sublayer_idx, results in all_results.items():
            sublayer = cfg.sublayers[sublayer_idx]
            tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
            group = f.create_group(tag)
            for name, array in results.items():
                group.create_dataset(name, data=array, compression="gzip")
            group.attrs["tag"] = tag
    print(f"Saved collision_opt results -> {cfg.collision_opt_results_path}")
