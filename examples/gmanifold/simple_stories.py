"""Embedding manifold of SimpleStories-V2-11M -> volume-uniform samples -> the
real block-0 FFN map -> validation against an independently fitted
destination manifold.

ONNX port of the original gmanifold repo's examples/simplestories.py: reads
real states via gmanifold.onnx_transformer (construct.construct +
onnxruntime) against configs/simple_stories.toml instead of live HF forward
hooks. Only "add_6" -> "add_8" (block-0's FFN sublayer) is usable as a
`make_map` destination here -- see gmanifold/onnx_transformer.py's docstring
for why sublayers crossing attention aren't supported by this ONNX approach.

Usage:
    uv run python -m examples.gmanifold.simple_stories --alpha 0.3
"""
import argparse

import torch

import gmanifold as gm
from config import load_config
from gmanifold import onnx_transformer as tr

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--dst", default="add_8", help="destination sublayer output tensor name")
ap.add_argument("--src", default="add_6", help="source sublayer input tensor name (must match --dst's declared sublayer)")
ap.add_argument("--latent-dim", type=int, default=16)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--config", default="configs/simple_stories.toml")
a = ap.parse_args()

cfg = load_config(a.config)

# 1. real states: X_in = vocabulary embeddings; X_out = the same tokens'
# residual at `dst`, read at sequence position 0 (the src->dst sublayer is
# tokenwise -- see onnx_transformer.make_map's docstring).
X_in, ids = tr.vocab_states(cfg)
input_ids = torch.zeros(len(ids), cfg.seq, dtype=torch.long)
input_ids[:, 0] = torch.tensor(ids)
positions = torch.zeros(len(ids), dtype=torch.long)
X_out = tr.collect_states(cfg, input_ids, positions, [a.src, a.dst])[a.dst]

perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
n_val = len(ids) // 10
tr_i, va_i = perm[:-n_val], perm[-n_val:]

# 2. two independent manifolds (source and destination); nothing about the model enters here
M_in = gm.fit(gm.GlobalManifold(latent_dim=a.latent_dim), X_in[tr_i], epochs=a.epochs, X_val=X_in[va_i])
M_out = gm.fit(gm.GlobalManifold(latent_dim=a.latent_dim), X_out[tr_i], epochs=a.epochs, X_val=X_out[va_i])

# 3. sample the source manifold, push through the real ONNX sublayer src -> dst, validate at both ends
X_sample, info = M_in.sample(a.n, alpha=a.alpha, seed=0)
map_fn = tr.make_map(cfg, a.src, a.dst)
Y_sample = map_fn(X_sample)


def show(title, rep):
    print(f"\n{title}\n" + "\n".join(
        f"  {k:20s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()) for k, v in rep.items()
    ))


T_in, T_out = gm.TangentCharts(X_in[tr_i], a.latent_dim), gm.TangentCharts(X_out[tr_i], a.latent_dim)
show(
    f"source (vocab embedding), alpha={a.alpha}, ESS={info['ess']:.0f}, coverage={M_in.coverage(X_sample):.2f}",
    gm.support_report(M_in, X_sample, X_in[va_i], gm.KernelScore(X_in[tr_i]), T_in),
)
X_tan, _ = T_in.sample(a.n, alpha=a.alpha, seed=0)  # conservative cross-check sampler
show(
    "source (vocab embedding) — tangent-chart samples (cross-check)",
    {"tangent samples": gm.support_report(M_in, X_tan, kernel=gm.KernelScore(X_in[tr_i]), tangent=T_in)["samples"]},
)
show(
    f"destination ({a.dst}) — images of the samples vs the independent destination fit",
    gm.support_report(M_out, Y_sample, X_out[va_i], gm.KernelScore(X_out[tr_i]), T_out),
)
show(
    f"destination ({a.dst}) — images of held-out REAL states (control)",
    {"real images": gm.support_report(M_out, map_fn(X_in[va_i]), kernel=gm.KernelScore(X_out[tr_i]))["samples"]},
)
