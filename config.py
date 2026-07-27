import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ONNX_DIR = Path("onnx_models")



@dataclass
class InputSourceConfig:
    mode: str = "random"
    tokenizer: str | None = None
    dataset: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    text_column: str = "text"
    norm: float | None = None
    norm_spread: float | None = None
    perturbation_spread: float | None = None

    def __post_init__(self):
        if self.mode not in ("random", "dataset", "fixed_norm", "characteristic"):
            raise ValueError(f"input_source.mode must be 'random', 'dataset', or 'fixed_norm', got {self.mode!r}")
        if self.mode == "dataset" and (self.tokenizer is None or self.dataset is None):
            raise ValueError("input_source.mode = 'dataset' requires 'tokenizer' and 'dataset'")
        if self.mode == "fixed_norm" and self.norm is None:
            raise ValueError("input_source.mode = 'fixed_norm' requires 'norm'")


@dataclass
class SublayerConfig:
    """One block of interest to extract: `input` is the upstream residual-stream tensor,
    `output` is the raw sublayer output (attn or ffn, post-norm, pre-residual-add) computed
    from it

    input_mean/input_min/input_max may be a single float (one baseline shared by every seq
    position -- the historical/seq=1 shape) or a list with one entry per seq position
    (0..cfg.seq-1), e.g. from generate_embedding_stats.py's per-position output. Use
    SublayerConfig.mean_at(position) etc. rather than reading the field directly, so callers
    don't need to handle both shapes themselves. None (the default) means "not set" -- the
    collisions pipeline auto-populates these by computing activation stats itself; see
    collisions/run_pipeline.py's ensure_activation_stats."""
    input: str
    output: str
    input_mean: float | list[float] | None = None
    input_min: float | list[float] | None = None
    input_max: float | list[float] | None = None

    @property
    def has_stats(self) -> bool:
        return self.input_mean is not None and self.input_min is not None and self.input_max is not None

    def _at(self, field: float | list[float] | None, position: int) -> float:
        assert field is not None, "stats not set -- see has_stats"
        return field[position] if isinstance(field, list) else field

    def mean_at(self, position: int) -> float:
        return self._at(self.input_mean, position)

    def min_at(self, position: int) -> float:
        return self._at(self.input_min, position)

    def max_at(self, position: int) -> float:
        return self._at(self.input_max, position)


PIPELINES = ("jacobians", "activations", "collisions", "collision_opt", "sublayer_jacobians")


@dataclass
class ModelConfig:
    name: str
    pipeline: str
    """Which orchestration pipeline this config drives: "jacobians" (convert_model.py +
    jacobians/run_pipeline.py), "activations" (embeddings/run_pipeline.py), or "collisions"
    (collisions/run_pipeline.py). A single config fully determines which one runs."""
    model: str
    input_name: str
    """Name of the model's forward input. "input_ids" feeds random token ids. Any other
    value (e.g. "inputs_embeds" for HF causal-LM models, to feed embeddings directly
    instead of token ids) feeds random floats of shape (batch, seq, input_dim) and
    requires `input_dim` to be set. "inputs_embeds" is additionally passed as a kwarg
    (not positionally) when exporting HF models, since it isn't forward()'s first
    positional argument."""
    vocab_size: int
    batch: int
    seq: int
    d_model: int
    needs_layernorm_patch: bool
    input_dim: int | None = None
    input_source: InputSourceConfig = field(default_factory=InputSourceConfig)
    jacobian_position: int | None = None
    """Sequence position to compute the diagonal Jacobian for (Python-style index).
    None (default) computes every position's diagonal block, like before. Set to e.g.
    -1 (last token) or 0 (first token) to backprop only that position's d_model
    outputs instead of every position's, for a cheaper single-position Jacobian."""
    sublayers: list[SublayerConfig] = field(default_factory=list)
    """Self-contained list of blocks of interest: each entry's (input, output) is also a
    severed-gradient pair (see `pairs`), used by convert_model.py/run_model.py's diagonal-
    Jacobian pipeline, and by extract_sub_graph in search_graphs.py for sub-graph
    extraction. Left empty for configs that don't need either."""
    svd: bool = False
    """jacobians pipeline only: save the SVD of each pair's diagonal-block Jacobians instead
    of the raw blocks."""
    slogdet_only: bool = False
    """jacobians pipeline only, requires `samples` to be set: instead of accumulating every
    sample's full (D_MODEL, D_MODEL) blocks in memory and saving once at the end (this
    config's default, and the only option when `samples` is unset), compute each sample's
    sign/log|det| right after that sample's pass and append it to
    diagonal_jacobians_path immediately, discarding the raw block. Peak memory no longer
    grows with `samples`, and a crash mid-run leaves every sample computed so far already on
    disk. Ignores `svd` (there's no SVD of a scalar determinant)."""
    samples: int | None = None
    """Number of samples to draw, shared across pipelines: activations pipeline uses it as
    the number of random-input forward passes to average activation stats (mean/min/max)
    over (defaults to 1000 if unset); jacobians pipeline uses it (if set) to draw that many
    independent inputs and compute each of jacobian_position's position(s) diagonal block for
    each input, stacking the results -- jacobian_position=None (default) samples every
    position, same as the non-sampling default. Unset `samples` (default) for jacobians
    computes one Jacobian from a single example, like before."""
    seed: int = 0
    """RNG seed for `samples`' random inputs (activations pipeline, and jacobians pipeline
    when `samples` is set)."""
    random_init: bool = False
    """Reinitialize every parameter with fresh random weights right after loading cfg.model
    (before ONNX export), instead of keeping its pretrained weights."""
    random_init_seed: int = 0
    """RNG seed for random_init's reinitialized weights."""
    save_dir: str | None = None
    """Where this config's h5 outputs (jacobians' diagonal_jacobians_path/svd_path,
    collisions' per-pair collision_distances.h5 files) are saved, instead of the default
    onnx_models/{name}/. Everything else (base model, gradient graph, subgraphs) still keys
    off `name` regardless of this field. Set it to reuse those results across configs that
    only differ in something like input_source's dataset -- rerunning with a new dataset
    won't overwrite results saved under a previous save_dir, or vice versa."""
    collision_opt_steps: int = 500
    """collision_opt pipeline only: number of gradient-descent steps to optimize y for."""
    collision_opt_lr: float = 1e-1
    """collision_opt pipeline only: learning rate for the y-optimization loop."""
    collision_opt_optimizer: str = "sgd"
    """collision_opt pipeline only: "adam" or "sgd", the update rule applied to y each step."""
    collision_opt_init_perturbation: float = 1e-2
    """collision_opt pipeline only: y0 = x + a random perturbation of this norm (via
    sample_ball_around_x), independent of input_source.mode."""
    collision_opt_position: int = -1
    """collision_opt pipeline only: seq position (Python-style index, like jacobian_position)
    to optimize the collision at; other seq slots are filled with the same sampled x/y as
    context, tiled identically, matching max_search.py's g() closure pattern."""
    collision_opt_loss: str = "plain"
    """collision_opt pipeline only: "plain" (sum((f(x)-f(y))**2)) or "normalized"
    (that, divided by sum((x-y)**2)) -- the loss y is gradient-descended on."""

    def __post_init__(self):
        if self.pipeline not in PIPELINES:
            raise ValueError(f"{self.name}: pipeline must be one of {PIPELINES}, got {self.pipeline!r}")
        if self.jacobian_position is not None and not (-self.seq <= self.jacobian_position < self.seq):
            raise ValueError(f"{self.name}: jacobian_position {self.jacobian_position} out of range for seq={self.seq}")
        if self.slogdet_only and self.samples is None:
            raise ValueError(f"{self.name}: slogdet_only requires samples to be set")
        if self.collision_opt_optimizer not in ("adam", "sgd"):
            raise ValueError(f"{self.name}: collision_opt_optimizer must be 'adam' or 'sgd', got {self.collision_opt_optimizer!r}")
        if not (-self.seq <= self.collision_opt_position < self.seq):
            raise ValueError(f"{self.name}: collision_opt_position {self.collision_opt_position} out of range for seq={self.seq}")
        if self.collision_opt_loss not in ("plain", "normalized"):
            raise ValueError(f"{self.name}: collision_opt_loss must be 'plain' or 'normalized', got {self.collision_opt_loss!r}")

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(s.input, s.output) for s in self.sublayers]

    @property
    def dir(self) -> Path:
        """This config's own subfolder -- every artifact it produces lives under here,
        except h5 outputs when save_dir is set -- see h5_dir."""
        return ONNX_DIR / self.name

    @property
    def h5_dir(self) -> Path:
        """Where this config's h5 outputs are saved: save_dir if set, else `dir` (the same
        per-config folder as everything else)."""
        return Path(self.save_dir) if self.save_dir is not None else self.dir

    @property
    def base_model_path(self) -> Path:
        return self.dir / "base_model.onnx"

    @property
    def per_pair_dir(self) -> Path:
        return self.dir / "per_pair_artifacts"

    @property
    def union_artifact_dir(self) -> Path:
        return self.dir / "union_artifacts"

    @property
    def gradient_path(self) -> Path:
        return self.dir / "gradient.onnx"

    @property
    def embeddings_path(self) -> Path:
        return self.dir / "embeddings.onnx"

    @property
    def diagonal_jacobians_path(self) -> Path:
        return self.h5_dir / "diagonal_blocks.h5"

    @property
    def sublayer_gradients_dir(self) -> Path:
        """sublayer_jacobians pipeline only: one independent per-pair training graph per
        sublayer (never unioned), keyed by f"sublayer{i}_{input}_{output}_gradient.onnx"."""
        return self.dir / "sublayer_jacobians"

    @property
    def sublayer_jacobians_dir(self) -> Path:
        """sublayer_jacobians pipeline only: each pair's own diagonal-block h5 result,
        keyed by f"sublayer{i}_{input}_{output}.h5" -- under save_dir if set, like
        diagonal_jacobians_path."""
        return self.h5_dir / "sublayer_jacobians"

    @property
    def svd_path(self) -> Path:
        return self.h5_dir / "svd_diagonal_blocks.h5"

    @property
    def collision_distances_dir(self) -> Path:
        return self.h5_dir / "collision_distances"

    @property
    def collision_opt_graph_dir(self) -> Path:
        return self.dir / "collision_opt_artifacts"

    @property
    def collision_opt_results_path(self) -> Path:
        return self.h5_dir / "collision_opt_results.h5"


def grad_names(cfg: ModelConfig) -> list[str]:
    """Per-pair grad output name for each pair's upstream tensor, following
    union_per_pair_artifacts.py's pivot-clash renaming: a pair's upstream keeps `{upstream}_grad`
    unless some earlier pair's downstream already produced a tensor of that same name (a pivot
    tensor) -- in which case ORT training's own `{tensor}_grad` name would collide with that
    earlier pair's loss-seed grad, so _resolve_pivot_clashes renames it to
    `pair{pair_idx}_{upstream}_grad` (pair_idx is 1-based, matching `models[1:]` in _merge_graphs).
    """
    names = []
    downstreams_seen: set[str] = set()
    for pair_idx, (upstream, downstream) in enumerate(cfg.pairs):
        if pair_idx > 0 and upstream in downstreams_seen:
            names.append(f"pair{pair_idx}_{upstream}_grad")
        else:
            names.append(f"{upstream}_grad")
        downstreams_seen.add(downstream)
    return names


def load_config(path: str) -> ModelConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    input_source = raw.pop("input_source", None)
    if input_source is not None:
        input_source = InputSourceConfig(**input_source)
    else:
        input_source = InputSourceConfig()
    sublayers = [SublayerConfig(**s) for s in raw.pop("sublayers", [])]
    return ModelConfig(**raw, input_source=input_source, sublayers=sublayers)