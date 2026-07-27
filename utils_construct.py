"""
Usage: uv run python3 convert_model.py configs/gpt2.toml
"""

import contextlib
import os
import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer


import onnx
import onnxruntime.training.artifacts as artifacts
import onnxruntime.training.onnxblock as onnxblock
import torch
from onnx import helper


from union_per_pair_artifacts import build_union_artifacts

from config import ModelConfig, grad_names, load_config



def _dummy_input(cfg: ModelConfig) -> torch.Tensor:
    if cfg.input_name == "input_ids":
        return torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq))
    assert cfg.input_dim is not None, f"{cfg.name}: input_dim required in config for non-input_ids models"
    return torch.randn(cfg.batch, cfg.seq, cfg.input_dim)


def _needs_embeds_kwarg(cfg: ModelConfig) -> bool:
    """`inputs_embeds` isn't the first positional arg in HF forward signatures (input_ids
    is), so it must be exported as a kwarg rather than a positional tensor -- otherwise it
    silently lands in e.g. `past_key_values` instead."""
    return cfg.input_name == "inputs_embeds" and ":" not in cfg.model

def _load_model(spec: str) -> torch.nn.Module:
    """`spec` is either an HF model id (passed to AutoModel.from_pretrained) or a
    "module:factory_fn" dotted path to a zero-arg local factory returning an eval() module.
    """
    if ":" in spec:
        module_name, factory_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        model = getattr(module, factory_name)()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(spec)
        model.eval()
        model.config.use_cache = False

    return model

def _random_init(model: torch.nn.Module, seed: int) -> None:
    """Reinitialize every parameter in-place with fresh random weights (matching each
    tensor's existing shape/dtype), instead of the pretrained/factory-built values
    _load_model produced -- works for any module (HF or custom factory) since it doesn't
    rely on architecture-specific init_weights()/reset_parameters() methods."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))

def export_base_model(cfg: ModelConfig) -> None:
    cfg.dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(cfg.model)
    if cfg.random_init:
        _random_init(model, cfg.random_init_seed)
    dummy_input = _dummy_input(cfg)
    if _needs_embeds_kwarg(cfg):
        args = ({"inputs_embeds": dummy_input},)
    else:
        args = (dummy_input,)
    torch.onnx.export(
        model, args, str(cfg.base_model_path), input_names=[cfg.input_name], output_names=["output"]
    )
    print(f"Saved base model -> {cfg.base_model_path}")


@contextlib.contextmanager
def exclusive_lock(lock_path: Path, poll_seconds: float = 2.0):
    """Waits until `lock_path` can be created exclusively (os.O_CREAT|O_EXCL is atomic even
    on network filesystems, unlike flock -- this repo's onnx_models/ can be reached from
    several cluster nodes at once), then holds it for the block, deleting it on exit (even
    on error, so a crash doesn't wedge future runs forever)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@contextlib.contextmanager
def chdir(path: Path):
    """Temporarily changes the process cwd. Every path passed to the wrapped block must be
    absolute -- callers are responsible for resolving any relative path first."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def ensure_base_model(cfg: ModelConfig, force: bool = False) -> None:
    """Exports cfg.base_model_path if it doesn't already exist (or always, if force), guarded
    by a lockfile so two processes sharing the same `name` (and therefore the same
    base_model_path -- e.g. several configs submitted as separate cluster jobs in parallel)
    never export/read the same file concurrently: onnx.load has no partial-write protection,
    so a reader can otherwise see another writer's half-written base_model.onnx.data.
    Whichever process gets the lock first exports; every other process waits, then finds the
    file already there (unless force) and skips its own export."""
    lock_path = cfg.base_model_path.with_suffix(cfg.base_model_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and cfg.base_model_path.exists():
            print(f"Base model already exists, skipping export -> {cfg.base_model_path}")
        else:
            print(f"Exporting base model ({cfg.model})...")
            export_base_model(cfg)


def _weight_dtype(model: onnx.ModelProto) -> int:
    """The TensorProto dtype of the model's weights, so patched-in inputs/outputs
    (mask, LayerNorm stats, layer values) match it instead of assuming fp32."""
    return model.graph.initializer[0].data_type


def _add_layer_norm_stats_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    # torch.onnx.export only wires up LayerNormalization's Y output. ORT training's
    # LayerNormalizationGrad builder unconditionally reads the Mean/InvStdDev outputs
    # (O(1)/O(2) in gradient_builder_base.h) with no guard, so any LayerNormalization
    # node without them crashes gradient graph building with:
    #   RuntimeError: ... i < node_->OutputDefs().size() was false.
    # https://github.com/microsoft/onnxruntime/issues/22955
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {
        vi.name: vi
        for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    dtype = _weight_dtype(model)

    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) > 1:
            continue
        x_vi = shapes.get(node.input[0])
        if x_vi is None or not x_vi.type.tensor_type.HasField("shape"):
            raise RuntimeError(f"No static shape for {node.input[0]!r} feeding {node.name}")
        x_shape = [d.dim_value for d in x_vi.type.tensor_type.shape.dim]
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        if axis < 0:
            axis += len(x_shape)
        stats_shape = x_shape[:axis] + [1] * (len(x_shape) - axis)

        mean_name, invstd_name = node.name + "_Mean", node.name + "_InvStdDev"
        node.output.extend([mean_name, invstd_name])
        for out_name in (mean_name, invstd_name):
            model.graph.value_info.append(
                helper.make_tensor_value_info(out_name, dtype, stats_shape)
            )

    # Not checked here: model's initializers are still external-data references (this
    # function runs on a load_external_data=False proto), and onnx.checker.check_model
    # can't resolve those against a bare in-memory proto -- it needs a path to resolve
    # relative to, same as _save_severable_model's own check_model(..., True) call. The
    # caller checks the saved file's path once it's on disk in the right directory instead.
    return model

def _prepare_severable_model(cfg: ModelConfig):
    """Patches the base model (LayerNorm stats outputs, `mask` input) and saves it to disk
    with external data, returning its path. generate_artifacts must be given a path rather
    than an in-memory ModelProto for models whose initializers exceed protobuf's 2GB message
    limit -- passing a ModelProto forces onnxblock to fully serialize the model in memory
    (SerializeToString), which silently overflows past 2GB instead of raising. A path is
    also required for onnxblock's internal check_model(..., True) calls to resolve the
    external-data location relative to the model's own directory instead of the process's
    cwd (the in-memory-proto code path has no directory to resolve against at all).
    """
    model = onnx.load(str(cfg.base_model_path), load_external_data = False)
    if cfg.needs_layernorm_patch:
        model = _add_layer_norm_stats_outputs(model)
    existing_inputs = {i.name for i in model.graph.input}
    if "mask" not in existing_inputs:
        # Always float32, regardless of the model's weight dtype: ORT's Python session.run
        # can't accept bfloat16 numpy inputs, and MaskedSumLoss casts `downstream` to float
        # before multiplying by `mask` so the dtypes match ONNX's Mul requirement.
        model.graph.input.append(
            helper.make_tensor_value_info("mask", onnx.TensorProto.FLOAT, [cfg.batch, cfg.seq, cfg.d_model])
        )

    return str(_save_severable_model(cfg,model))


def _save_severable_model(cfg: ModelConfig, model: onnx.ModelProto) -> Path:
    """Saves the patched model to its own file (never cfg.base_model_path), alongside
    cfg.base_model_path in cfg.dir, so onnxblock's check_model(..., True) can resolve the
    external-data location relative to that directory (where the base model's .onnx.data
    already lives) instead of the process's cwd. Also runs onnx.checker.check_model on the
    saved path (not the in-memory proto) for the same reason -- the proto's initializers are
    still external-data references, which the checker can only resolve given a path to
    resolve them relative to."""
    severable_model_path = cfg.dir / "severable.onnx"
    onnx.save(
        model,
        str(severable_model_path)
    )
    onnx.checker.check_model(str(severable_model_path))
    return severable_model_path


def _add_layer_value_outputs(model: onnx.ModelProto, layer_names: list[str], cfg: ModelConfig) -> onnx.ModelProto:
    """Expose each severed layer's own activations as a graph output (not just its
    gradient), so callers can read e.g. `add_1` alongside `add_1_grad`. generate_artifacts
    keeps the full forward pass regardless of which slice has requires_grad, so every layer
    name is already a node output in every per-pair graph -- it just isn't declared yet.
    """
    existing = {o.name for o in model.graph.output}
    dtype = _weight_dtype(model)
    for name in layer_names:
        if name not in existing:
            model.graph.output.append(
                helper.make_tensor_value_info(name, dtype, [cfg.batch, cfg.seq, cfg.d_model])
            )
    return model

def _set_temp_file_name(block: onnxblock.blocks.Block, temp_dir: Path, temp_file_name: str) -> onnxblock.blocks.Block:
    # Block.__init__ hardcodes "temp.onnx" for every instance, so sibling blocks called
    # back-to-back under the same has_path build collide on temp.onnx.data (onnx.save
    # refuses to overwrite it). Give each block its own scratch file name instead, under
    # temp_dir (cfg.union_artifact_dir -- per-config, not the shared process cwd): two
    # configs/jobs running at once can otherwise collide on an identical tag (e.g. two GPT-2
    # variants both building "pair0_add_1_add_4"), and clear_stale_temp_files's cleanup
    # sweep would delete a concurrent job's in-progress scratch file if it swept a shared dir.
    block.temp_onnx_file_path = str(temp_dir / temp_file_name)
    block.temp_external_data_file_name = temp_file_name + ".data"
    return block

class MaskedSumLoss(onnxblock.blocks.Block):
    """loss = sum(downstream * mask). One per per-pair graph.

    ORT training's ReduceSumGrad/Expand hardcodes tensor(float) for the loss's gradient
    seed, so bf16 (or other non-fp32) forward graphs must be cast to float before
    ReduceSum -- otherwise gradient graph building fails with a type mismatch on the
    ReduceSum grad node. `downstream` is cast to float first (rather than casting the
    Mul's output), so `mask` itself is always plain float32: ONNX's Mul requires both
    operands to share a dtype, and ORT's Python session.run can't accept bfloat16 numpy
    arrays as inputs at all (ml_dtypes' bfloat16 isn't a supported input type), so `mask`
    can never be bf16 in practice. Cast has a well-defined gradient (cast back), so
    upstream gradients still flow through the rest of the model in its original dtype.
    """

    def __init__(self, tag: str, temp_dir: Path):
        super().__init__()
        _set_temp_file_name(self, temp_dir, f"temp_loss_{tag}.onnx")
        self._cast = _set_temp_file_name(onnxblock.blocks.Cast(onnx.TensorProto.FLOAT), temp_dir, f"temp_cast_{tag}.onnx")
        self._mul = _set_temp_file_name(onnxblock.blocks.Mul(), temp_dir, f"temp_mul_{tag}.onnx")
        self._reduce_sum = _set_temp_file_name(onnxblock.blocks.ReduceSum(keepdims=False), temp_dir, f"temp_reduce_sum_{tag}.onnx")

    def build(self, downstream_name):
        
        downstream_f32 = self._cast(downstream_name)
        masked = self._mul(downstream_f32, "mask")
        return self._reduce_sum(masked)
    
def _add_f32_casts(model: onnx.ModelProto, existing: set[str]) -> None:
    """For every one of `model`'s current graph outputs that isn't float32, appends an
    in-graph Cast to float32 and makes the *cast* tensor a graph output too (in place),
    mutating `existing` to track newly-added names.

    ORT's Python bindings can never convert a bf16 tensor to numpy (session.run and
    OrtValue.numpy() both raise "No corresponding Numpy type for Tensor Type" -- there's no
    io_binding workaround, it's baked into the compiled extension), so every bf16 output
    needs this cast counterpart to ever be fetched from a Python session.run call.
    """
    for o in list(model.graph.output):
        if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            continue
        out_name = f"{o.name}_f32"
        if out_name in existing:
            continue
        model.graph.node.append(helper.make_node("Cast", [o.name], [out_name], to=onnx.TensorProto.FLOAT))
        model.graph.output.append(helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None))
        existing.add(out_name)


def _add_grad_outputs(cfg: ModelConfig, union_model_path: Path) -> None:
    """Appends each pair's upstream-gradient as a graph output, in place.

    grad_name's dtype matches the model's weight dtype (e.g. bf16): it's the gradient of
    `upstream`, a bf16 activation, and MaskedSumLoss casts `downstream` to float only for
    the loss/mask math -- gradients flowing back through the rest of the (bf16) model stay
    bf16.
    """
    model = onnx.load(str(union_model_path))
    existing = {o.name for o in model.graph.output}
    weight_dtype = _weight_dtype(model)

    _add_f32_casts(model, existing)

    for grad_name in grad_names(cfg):
        if grad_name not in existing:
            model.graph.output.append(
                helper.make_tensor_value_info(grad_name, weight_dtype, [cfg.batch, cfg.seq, cfg.d_model])
            )
            existing.add(grad_name)

    _add_f32_casts(model, existing)
    onnx.save(model, str(union_model_path))


def build_per_pair_artifacts(cfg: ModelConfig) -> list[Path]:
    cfg.union_artifact_dir.mkdir(parents=True, exist_ok=True)
    severable_model = _prepare_severable_model(cfg)
    severable_model_abs = str(Path(severable_model).resolve())
    union_dir_abs = cfg.union_artifact_dir.resolve()
    paths = []
    for pair_idx, (upstream, downstream) in enumerate(cfg.pairs):
        tag = f"pair{pair_idx}_{upstream}_{downstream}"

        # onnxblock's internal _TrainingBlock (built inside generate_artifacts, not
        # something we can pass a temp_dir into) falls back to Block.__init__'s default
        # temp_onnx_file_path = os.getcwd()/temp.onnx whenever it saves+checks the model
        # itself (e.g. infer_shapes_on_base). By that point the model's tensors may already
        # carry external-data locations set relative to cfg.union_artifact_dir (from our own
        # MaskedSumLoss/_cast/_mul/_reduce_sum blocks, whose temp_onnx_file_path IS patched
        # via _set_temp_file_name) -- saving/checking from a different cwd then looks for
        # those filenames in the wrong directory and onnx.checker fails with "is not a
        # regular file" even though the data is present. chdir into union_artifact_dir so
        # the unpatched default resolves to the same directory as everything else.
        with chdir(union_dir_abs):
            artifacts.generate_artifacts(
                severable_model_abs,
                requires_grad=[upstream],
                loss_input_names=[downstream],
                prefix=f"{tag}_",
                loss=MaskedSumLoss(tag, union_dir_abs),
                artifact_directory=str(union_dir_abs),
            )
        training_model_path = cfg.union_artifact_dir / f"{tag}_training_model.onnx"
        layer_names = sorted({name for pair in cfg.pairs for name in pair})
        training_model = onnx.load(str(training_model_path))
        training_model = _add_layer_value_outputs(training_model, layer_names, cfg)
        onnx.save(
            training_model,
            str(training_model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{tag}_training_model.onnx.data",
        )

        paths.append(training_model_path)



    return paths

def sublayer_gradient_path(cfg: ModelConfig, sublayer_idx: int) -> Path:
    sublayer = cfg.sublayers[sublayer_idx]
    tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
    return cfg.sublayer_gradients_dir / f"{tag}_gradient.onnx"


def build_single_pair_gradient_graph(cfg: ModelConfig, sublayer_idx: int, force: bool = False) -> Path:
    """Builds one sublayer's own independent gradient graph -- same per-pair ORT-training
    artifact build_per_pair_artifacts uses for each pair on its way into the union graph,
    except this one is never merged with any other pair's graph and its grad output keeps
    the plain `{upstream}_grad` name (no pivot-clash renaming -- there's only one pair, so
    no other pair's tensor can ever collide with it).

    Guarded by a lockfile for the same reason build_gradient_graph needs one: two processes
    sharing the same `name` could otherwise race on this pair's own artifact directory.
    """
    out_path = sublayer_gradient_path(cfg, sublayer_idx)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        if not force and out_path.exists():
            print(f"Gradient graph already exists, skipping build -> {out_path}")
            return out_path

        sublayer = cfg.sublayers[sublayer_idx]
        upstream, downstream = sublayer.input, sublayer.output
        tag = f"sublayer{sublayer_idx}_{upstream}_{downstream}"
        build_dir = cfg.sublayer_gradients_dir / f"{tag}_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        severable_model = _prepare_severable_model(cfg)
        severable_model_abs = str(Path(severable_model).resolve())
        build_dir_abs = build_dir.resolve()

        with chdir(build_dir_abs):
            artifacts.generate_artifacts(
                severable_model_abs,
                requires_grad=[upstream],
                loss_input_names=[downstream],
                prefix=f"{tag}_",
                loss=MaskedSumLoss(tag, build_dir_abs),
                artifact_directory=str(build_dir_abs),
            )
        training_model_path = build_dir / f"{tag}_training_model.onnx"
        training_model = onnx.load(str(training_model_path))
        training_model = _add_layer_value_outputs(training_model, [upstream, downstream], cfg)

        existing = {o.name for o in training_model.graph.output}
        weight_dtype = _weight_dtype(training_model)
        _add_f32_casts(training_model, existing)
        grad_name = f"{upstream}_grad"
        if grad_name not in existing:
            training_model.graph.output.append(
                helper.make_tensor_value_info(grad_name, weight_dtype, [cfg.batch, cfg.seq, cfg.d_model])
            )
            existing.add(grad_name)
        _add_f32_casts(training_model, existing)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(
            training_model, str(out_path),
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=f"{tag}_gradient.onnx.data",
        )
        shutil.rmtree(build_dir)

    return out_path


def clear_stale_temp_files(cfg: ModelConfig) -> None:
    """Block.__init__ (and _set_temp_file_name) scratch files from a prior run/attempt --
    onnx.save refuses to overwrite an existing external-data file, so leftovers from a
    failed or --force-repeated run must be cleared before building starts. Scoped to this
    config's own cfg.union_artifact_dir (where _set_temp_file_name now places them), not the
    shared process cwd -- a cwd-wide sweep would delete another config/job's in-progress
    scratch files if two builds ran concurrently in the same checkout."""
    if not cfg.union_artifact_dir.exists():
        return
    for stale in cfg.union_artifact_dir.glob("temp*.onnx*"):
        stale.unlink()


def extract_sub_graph(cfg: ModelConfig) -> list[Path]:
    cfg.per_pair_dir.mkdir(parents=True, exist_ok=True)

    base_model = onnx.load(str(cfg.base_model_path))
    base_initializers = {i.name: i for i in base_model.graph.initializer}

    paths = []
    for sublayer_idx, sublayer in enumerate(cfg.sublayers):
        tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
        sub_graph_path = cfg.per_pair_dir / f"{tag}_subgraph_eval.onnx"
        onnx.utils.extract_model(
            str(cfg.base_model_path),
            str(sub_graph_path),
            input_names=[sublayer.input],
            output_names=[sublayer.output],
        )
        paths.append(sub_graph_path)
    return paths

def ensure_sub_graphs(cfg: ModelConfig, force: bool = False) -> list[Path]:
    """Extracts cfg.per_pair_dir's subgraphs if they don't already all exist (or always, if
    force), guarded by a lockfile for the same reason ensure_base_model needs one: two
    configs sharing a `name` (and therefore cfg.per_pair_dir) submitted as separate parallel
    jobs would otherwise race on the same subgraph files -- one job's onnx.utils.extract_model
    write can land while another job's onnx.checker.check_model read is in flight, corrupting
    the read with a partial/invalid protobuf."""
    lock_path = cfg.per_pair_dir.with_name(cfg.per_pair_dir.name + ".lock")
    paths = [
        cfg.per_pair_dir / f"sublayer{i}_{s.input}_{s.output}_subgraph_eval.onnx"
        for i, s in enumerate(cfg.sublayers)
    ]
    with exclusive_lock(lock_path):
        if not force and all(p.exists() for p in paths):
            print(f"Sub-graphs already exist, skipping extraction -> {cfg.per_pair_dir}")
            return paths
        print("Extracting sub-graphs...")
        return extract_sub_graph(cfg)