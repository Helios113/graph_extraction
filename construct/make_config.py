"""Generate a model config TOML (see configs/*.toml, construct/config.py) from a Hugging Face model id.

    uv run python -m construct.make_config openai-community/gpt2 --name gpt2 --seq 4 --out configs/gpt2.toml

The boundary tensor names (the `input`/`output` of each sublayer) are the opaque names the static-shape
`torch.onnx.export` gives the residual stream (`add`, `add_2`, ...). They are found, not guessed: forward
hooks capture the residual stream in torch -- the embedding output, each decoder layer's post-attention
residual (layer input + attention output) and its output -- and each is matched by value against every
intermediate tensor of the exported graph. The chain is `input_ids -> embedding -> post-attn(0) -> post-MLP(0)
-> ...` (2 sublayers per layer + 1), as in configs/gpt2.toml.

Supports sequential pre-norm residual decoders (GPT-2, Llama/Qwen/Mistral, OPT, GPT-Neo, ...). Architectures whose
residual stream is not `x + attn(norm(x))` then `+ mlp(norm(.))` (parallel residual such as GPT-NeoX/Pythia/GPT-J,
sandwich norms such as Gemma) have no such boundary tensors; the script then exits with an error naming the layer
that failed to match instead of writing a wrong config.

Weights are randomly initialised from the HF config (no weight download): the graph, and so the tensor names, depend
only on the architecture. The emitted `model` is the HF id, so ensure_base_model exports the pretrained weights.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForCausalLM

LOSS_TYPE = "JacobianExtractorLoss"


def find_layers(model, n_layers: int) -> tuple[str, torch.nn.ModuleList]:
    """The decoder-block ModuleList: the (largest) ModuleList of length num_hidden_layers."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers:
            if best is None or sum(p.numel() for p in mod.parameters()) > sum(p.numel() for p in best[1].parameters()):
                best = (name, mod)
    if best is None:
        raise SystemExit(f"no ModuleList with {n_layers} decoder layers found")
    return best


def find_attention(layer: torch.nn.Module) -> torch.nn.Module:
    for name, child in layer.named_children():
        if "attn" in name or "attention" in name:
            return child
    raise SystemExit(f"no attention submodule in layer children {[n for n, _ in layer.named_children()]}")


def first_tensor(x):
    return x[0] if isinstance(x, tuple) else x


def residual_stream(model, layers):
    """Torch-side boundary values: [embedding, post-attn(0), post-MLP(0), post-attn(1), ...]."""
    cap: dict = {}
    hooks = [layers[0].register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("emb", (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)]
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, kw, i=i: cap.__setitem__(("in", i), (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True))
        hooks.append(find_attention(layer).register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("attn", i), first_tensor(o).detach())))
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("out", i), first_tensor(o).detach())))
    return cap, hooks


def make_config(hf_id: str, name: str, batch: int, seq: int, data_batch: int, trust_remote_code: bool = False) -> str:
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    n_layers, d_model, vocab = cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code, torch_dtype=torch.float32).eval()
    model.config.use_cache = False
    layer_path, layers = find_layers(model, n_layers)
    ids = torch.randint(0, vocab, (2, seq))
    cap, hooks = residual_stream(model, layers)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    want = [("embedding", cap["emb"])]
    for i in range(n_layers):
        want += [(f"post-attn({i})", cap[("in", i)] + cap[("attn", i)]), (f"post-MLP({i})", cap[("out", i)])]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.onnx"
        torch.onnx.export(model, (ids,), str(path), input_names=["input_ids"], output_names=["output"])
        g = onnx.load(str(path))
        producer = {o: n for n in g.graph.node for o in n.output}
        order = [o for n in g.graph.node for o in n.output]
        g.graph.output.extend([onnx.ValueInfoProto(name=o) for o in order if o != "output"])
        sess = ort.InferenceSession(g.SerializeToString(), providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        vals = dict(zip(names, sess.run(names, {"input_ids": ids.numpy()})))

    tensors = []
    for label, ref in want:
        ref = ref.numpy()
        tol = 1e-4 * max(1.0, float(np.abs(ref).max()))
        cands = [o for o in order if o in vals and vals[o].shape == ref.shape and np.abs(vals[o] - ref).max() <= tol]
        cands = [o for o in cands if producer[o].op_type not in ("Identity", "Reshape", "Cast")] or cands
        if not cands:
            raise SystemExit(
                f"{hf_id}: no ONNX tensor matches the {label} residual-stream value. This architecture's residual stream is not "
                "x + attn(norm(x)) then + mlp(norm(.)) (parallel residual / sandwich norms?), so it has no such boundary.",
            )
        tensors.append(cands[0])  # first producer (the Add / Gather itself, not a later copy)
    if len(set(tensors)) != len(tensors):
        raise SystemExit(f"{hf_id}: boundary tensors are not distinct: {tensors}")

    lines = [
        f'name = "{name}"',
        f'model = "{hf_id}"   # HF model id, passed to AutoModelForCausalLM.from_pretrained',
        'input_name = "input_ids"',
        f"vocab_size = {vocab}",
        f"batch = {batch}                   # Jacobian one-hot sweep is chunked in batches of this size",
        f"seq = {seq}",
        f"d_model = {d_model}",
        f'block_prefix = "/{layer_path.replace(".", "/")}"   # decoder-block name prefix in the optimum KV-cache export',
        f"network_dims = [{(d_model + 128) // 2}, 128]   # autoencoder hidden/latent dims after d_model",
        "",
        "[input_source]",
        'tokenizer = "SimpleStories/SimpleStories-V2-11M"',
        'dataset = "Salesforce/wikitext"',
        'dataset_config = "wikitext-2-raw-v1"',
        'split = "train"',
        'text_column = "text"',
        "",
    ]
    for a, b in zip(["input_ids"] + tensors[:-1], tensors):
        lines += ["", "[[sublayers]]", f'input = "{a}"', f'output = "{b}"', f'loss_type = "{LOSS_TYPE}"', f"input_shape = [{data_batch}, {seq}, {d_model}]"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="Hugging Face model id")
    ap.add_argument("--name", help="config name (default: last path part of the id, lower-cased)")
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024, help="`batch` field: Jacobian chunk size")
    ap.add_argument("--data-batch", type=int, default=32, help="leading dim of each sublayer's input_shape")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", type=Path, help="output path (default: configs/<name>.toml)")
    a = ap.parse_args()
    name = a.name or a.model.split("/")[-1].lower().replace(".", "_")
    out = a.out or Path("configs") / f"{name}.toml"
    toml = make_config(a.model, name, a.batch, a.seq, a.data_batch, a.trust_remote_code)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(toml)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
