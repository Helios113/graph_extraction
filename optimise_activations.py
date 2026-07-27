import h5py
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from tqdm import tqdm
from scipy.stats import truncnorm

from config import ModelConfig, InputSourceConfig

NUM_REFINE_ROUNDS = 50
PERTURBATION_SCALE = 1e-1


def _load_io_session(sub_graph_path):
    model = onnx.load(str(sub_graph_path))
    _wrap_bfloat16_io_with_float32(model)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def _wrap_bfloat16_io_with_float32(model: onnx.ModelProto) -> None:
    """Severed sub-graphs declare bf16 input/output (Qwen2's native dtype) -- ORT's Python
    session.run can't accept or return bf16 numpy arrays at all, so the graph's exposed I/O
    must be float32. Rather than casting the whole graph to float32 (losing genuine bf16
    compute), insert a Cast node at each bf16 boundary and leave every internal tensor bf16."""
    graph = model.graph

    for inp in list(graph.input):
        if inp.type.tensor_type.elem_type != onnx.TensorProto.BFLOAT16:
            continue
        bf16_name = inp.name + "_bf16"
        cast_node = helper.make_node(
            "Cast", inputs=[inp.name], outputs=[bf16_name],
            to=onnx.TensorProto.BFLOAT16, name=f"{inp.name}_postcast",
        )
        graph.node.insert(0, cast_node)
        for node in graph.node[1:]:
            node.input[:] = [bf16_name if x == inp.name else x for x in node.input]
        inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for out in list(graph.output):
        if out.type.tensor_type.elem_type != onnx.TensorProto.BFLOAT16:
            continue
        original_name = out.name
        renamed_bf16_name = out.name + "_bf16"
        for node in graph.node:
            node.output[:] = [renamed_bf16_name if x == original_name else x for x in node.output]
        cast_node = helper.make_node(
            "Cast", inputs=[renamed_bf16_name], outputs=[original_name],
            to=onnx.TensorProto.FLOAT, name=f"{original_name}_precast",
        )
        graph.node.append(cast_node)
        out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT





def sample_points(cfg: ModelConfig, sublayer_idx: int, position: int, seed: int) -> np.ndarray:
    src = cfg.input_source
    if src.mode == "fixed_norm":
        assert src.norm is not None
        spread = src.norm_spread if src.norm_spread is not None else src.norm / 100
        vmin, vmax = max(0.0, src.norm - spread), src.norm + spread
        return _sample_points_with_specific_distribution(cfg.batch, cfg.d_model, src.norm, vmin, vmax, seed)
    elif src.mode == "characteristic":
        stats = cfg.sublayers[sublayer_idx]
        return _sample_points_with_specific_distribution(cfg.batch, cfg.d_model, stats.mean_at(position), stats.min_at(position), stats.max_at(position), seed)
    raise ValueError

def _output_norms(session: ort.InferenceSession, input_name: str, output_name: str, candidates: np.ndarray) -> np.ndarray:
    """Runs `candidates` (batch, seq, d_model) fp32 through the session and returns the
    output's per-token (d_model) norm as (batch, seq) -- the seq axis is kept (not flattened)
    since attention's causal mixing makes a position's output distribution depend on how much
    context precedes it; the FFN sublayer is token-independent so its per-position norms are
    all equivalent and can be pooled by the caller if desired."""
    feed = {input_name: candidates.astype(np.float32)}
    (out,) = session.run([output_name], feed)
    return np.linalg.norm(out, axis=-1)


def sample_ball_around_x(x: np.ndarray, r: float, n_samples: int, seed=None) -> np.ndarray:
    """
    Uniform samples (by volume) inside the ball of radius 2r centered at each x.
    x: shape (N, d)
    Returns: shape (N, d)
    """
    d = x.shape[-1]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_samples, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # radius scaling: 2 * r * U gives uniform density over volume
    u = rng.uniform(size=n_samples)
    radii = r * u
    return x[:, None] + directions * radii[:, None]

def _sample_points_with_specific_distribution(n_vectors: int, dim: int,
                                mean: float, vmin: float, vmax: float, seed:int = 0) -> np.ndarray:
    """
    Returns array of shape (n_vectors, dim) such that the norms of the
    rows follow a truncated normal with given mean, min, max.
    """
    if not (0 <= vmin <= mean <= vmax):
        raise ValueError("need 0 <= vmin <= mean <= vmax")
    std = min(mean - vmin, vmax - mean) / 3
    rng = np.random.default_rng(seed)
    a, b = (vmin - mean) / std, (vmax - mean) / std
    norms = truncnorm.rvs(a, b, loc=mean, scale=std, size=n_vectors, random_state=rng)

    directions = rng.normal(size=(n_vectors, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    vectors = directions * norms[:, None]
    return vectors.reshape(n_vectors, dim)



def compute_collision_distances(
    cfg: ModelConfig,
    sublayer_idx: int,
    g_max: float,
    position: int,
    num_y: int = 10_000,
    seed: int = 0,
):
    """Restricts the collision search to output position `position` (0..cfg.seq-1): a
    position's output distribution can depend on how much causal context precedes it (true
    for attention; harmless but redundant for the token-independent FFN, whose positions are
    all equivalent), so pooling positions together would mix distributions that aren't
    comparable. Context positions preceding `position` are filled with the same
    representative-points sampling used for `position` itself (independent per call, not
    reused across positions)."""

    sublayer = cfg.sublayers[sublayer_idx]
    tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}_pos{position}"
    sub_graph_path = cfg.per_pair_dir / f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}_subgraph_eval.onnx"
    session = _load_io_session(sub_graph_path)
    run_input_name, run_output_name = sublayer.input, sublayer.output

    def g(points: np.ndarray) -> np.ndarray:
        """points: (cfg.batch, d_model) for the target position, tiled across every seq slot
        (context slots get independent draws of the same distribution -- see docstring)
        before running the session, then sliced back down to just `position`'s output."""
        tiled = np.tile(points[:, None, :], (1, cfg.seq, 1))
        feed = {run_input_name: tiled.astype(np.float32)}
        (out,) = session.run([run_output_name], feed)
        return out[:, position, :]

    x = sample_x(cfg, sublayer_idx, position, seed)
    f_x = x + g(x)

    distances_image = np.empty(num_y, dtype=np.float32)
    distances_domain = np.empty(num_y, dtype=np.float32)

    for start in tqdm(range(0, num_y, cfg.batch), desc=f"{tag} collision search"):
        y = sample_ball_around_x(x, g_max, cfg.batch)
        f_y = y + g(y)
        distances_image[start:start + cfg.batch] = np.linalg.norm(f_y - f_x, axis=-1).reshape(-1)
        distances_domain[start:start + cfg.batch] = np.linalg.norm(y - x, axis=-1).reshape(-1)

    float32_eps = np.finfo(np.float32).eps
    n_near_zero = int(np.count_nonzero(distances_image <= float32_eps))

    data_path = cfg.collision_distances_dir / f"{tag}_collision_distances.h5"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "w") as f:
        f.create_dataset("distances_domain", data=distances_domain, compression="gzip")
        f.create_dataset("distances_image", data=distances_image, compression="gzip")
        f.attrs["tag"] = tag
        f.attrs["g_max"] = g_max
        f.attrs["num_y"] = num_y
        f.attrs["float32_eps"] = float32_eps
        f.attrs["n_near_zero"] = n_near_zero
    print(f"{tag}: saved collision distances to {data_path}")


def compute_max_norm(
    cfg: ModelConfig,
    disable_progress: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """For each sublayer and each seq position, searches the sphere of radius x_norm in
    d_model space for the input that maximizes that position's sublayer-subgraph output
    norm. Candidates fill every seq slot identically (so the searched point also serves as
    its own causal context) and are scored only at the target position -- broken out per
    position rather than pooled across cfg.batch*cfg.seq, since attention's causal mixing
    makes a position's output distribution depend on how much context precedes it (the FFN
    sublayer is token-independent, so its positions are all equivalent, but it's still
    reported per position here for a uniform, directly comparable output). Batched random
    search (cfg.batch candidates per ORT call) followed by local hill-climbing refinement
    around the best candidate found so far. Returns {"{sublayer_tag}_pos{position}": max
    output norm found}.

    x_norm defaults to sqrt(d_model), matching compute_ffn_operator_norm_bound's default (the
    natural scale for a pre-RMSNorm residual-stream vector).

    The collision-search perturbation radius passed to compute_collision_distances is this
    searched max output norm, unless input_source.mode == "fixed_norm", in which case it's
    input_source.perturbation_spread instead (the searched best_norm is irrelevant once x
    itself is pinned to input_source.norm).
    """
    x_norm = float(np.sqrt(cfg.d_model))
    samples = cfg.samples if cfg.samples is not None else 10_000
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    for sublayer_idx, sublayer in enumerate(cfg.sublayers):
        base_tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
        sub_graph_path = cfg.per_pair_dir / f"{base_tag}_subgraph_eval.onnx"
        session = _load_io_session(sub_graph_path)
        run_input_name, run_output_name = sublayer.input, sublayer.output

        for position in range(cfg.seq):
            tag = f"{base_tag}_pos{position}"

            candidates = sample_x(cfg, sublayer_idx, position, seed)
            tiled = np.tile(candidates[:, None, :], (1, cfg.seq, 1))

            norms = _output_norms(session, run_input_name, run_output_name, tiled)[:, position]
            best_idx = int(np.argmax(norms))
            best_point = candidates[best_idx]
            best_norm = float(norms[best_idx])

            if cfg.input_source.mode != "fixed_norm":
                # The refine loop hill-climbs over the whole x_norm = sqrt(d_model) sphere,
                # which is the wrong scale (and pointless extra work) once x itself is
                # pinned to input_source.norm -- skip it in that mode.
                for _ in tqdm(range(NUM_REFINE_ROUNDS), desc=f"{tag} refine", disable=disable_progress):
                    # Perturb the current best token (every batch slot gets an independent
                    # perturbation of the same starting point, tiled identically across every
                    # seq slot), and keep whichever candidate wins -- zeroth-order hill-climbing
                    # since the ORT eval session has no gradients.
                    noise = rng.standard_normal((cfg.batch, cfg.d_model)) * x_norm
                    perturbed = best_point[None, :] + noise
                    perturbed = perturbed / np.linalg.norm(perturbed, axis=-1, keepdims=True) * PERTURBATION_SCALE * x_norm
                    perturbed_tiled = np.tile(perturbed[:, None, :], (1, cfg.seq, 1))
                    round_norms = _output_norms(session, run_input_name, run_output_name, perturbed_tiled)[:, position]
                    round_best_idx = int(np.argmax(round_norms))
                    if round_norms[round_best_idx] > best_norm:
                        best_norm = float(round_norms[round_best_idx])
                        best_point = perturbed[round_best_idx]

            results[tag] = best_norm
            print(f"{tag}: max output norm = {best_norm:.4f}")
            if cfg.input_source.mode == "fixed_norm":
                g_max = cfg.input_source.perturbation_spread
            else:
                g_max = best_norm
            compute_collision_distances(cfg, sublayer_idx, g_max, position, num_y=samples)

    return results
