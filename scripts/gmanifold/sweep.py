"""Sampling-centric gmanifold sweep: for each requested residual location, fit
a GlobalManifold, report an alpha sweep of the samples (nearest-real, recon,
kernel u, coverage, ESS) next to held-out/noise calibration bands, and (for
tokenwise "src.dst" location pairs -- see gmanifold/onnx_transformer.py) push
source-location samples through the real ONNX sublayer and validate the
images against an independently fitted destination manifold.

ONNX port of the original gmanifold repo's experiments/sweep.py: real states
are read via gmanifold.onnx_transformer (construct.construct + onnxruntime)
against a ModelConfig toml instead of live HF forward hooks.

Resilience, per CLAUDE.md's pattern for long-running sweeps over a config's
sublayers: one JSON report file per location under --out-dir (not h5 -- a
manifold's summary report is a handful of scalars/dicts per (location,
latent_dim, seed, alpha), not a huge row-wise dataset, so a small per-location
JSON is the natural unit here rather than forcing ShardedLayerWriter's
row-sharded h5 schema onto it). Each location's report is written and pushed
to the remote store as soon as that location's fit+sweep finishes, so a run
that dies partway through only redoes the in-progress location -- re-running
the same command skips every location whose report file already exists.

Usage:
    uv run python -m scripts.gmanifold.sweep configs/simple_stories.toml \
        --out-dir onnx_models/simple_stories/gmanifold_sweep \
        --locs embedding add_6 add_8
"""
import argparse
import json
import os
import time

import torch

import gmanifold as gm
from config import load_config
from execute.remote_storage import push
from gmanifold import onnx_transformer as tr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _report_path(out_dir: str, loc: str) -> str:
    return os.path.join(out_dir, f"{loc}.json")


def _write_report(out_dir: str, loc: str, report: dict, no_push: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = _report_path(out_dir, loc)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp_path, path)  # atomic on the same filesystem
    if not no_push:
        push(path)


def run_location(
    cfg,
    loc: str,
    X: torch.Tensor,
    tr_i: torch.Tensor,
    va_i: torch.Tensor,
    latent_dims: list[int],
    seeds: list[int],
    alphas: list[float],
    n_samples: int,
    epochs: int,
) -> dict:
    Xtr, Xva = X[tr_i].to(DEVICE), X[va_i].to(DEVICE)
    dim = gm.intrinsic_dimension(Xtr)
    kernel = gm.KernelScore(Xtr)
    report = dict(loc=loc, dim=dim, fits=[], alpha=[])
    for latent_dim in latent_dims:
        T = gm.TangentCharts(Xtr, latent_dim)
        for seed in seeds:
            t0 = time.time()
            M = gm.fit(gm.GlobalManifold(latent_dim=latent_dim, device=DEVICE), Xtr, epochs=epochs, X_val=Xva, seed=seed)
            jr = M.jacobian_rank()
            report["fits"].append(dict(
                m=latent_dim, seed=seed, spacing_unit=M.spacing_unit, hub_share=M.hub_share,
                val_recon=M.history[-1]["val_recon_over_spacing"], train_recon=M.history[-1]["recon"],
                rank=jr["rank_median"], jac_sv=jr["singular_values"][:latent_dim].tolist(),
                seconds=time.time() - t0,
            ))
            bands = gm.support_report(M, Xva, Xva, kernel, T)  # held-out real, 0.5x, 1x noise
            for alpha in alphas:
                x, info = M.sample(n_samples, alpha=alpha, auto_tau=True, seed=seed)
                rep = gm.support_report(M, x, kernel=kernel, tangent=T)["samples"]
                report["alpha"].append(dict(
                    m=latent_dim, seed=seed, alpha=alpha, ess=info["ess"], tau=info["tau"],
                    anchor_participation=info["anchor_participation"],
                    multiplicity=float(info["multiplicity"].float().mean()),
                    coverage=M.coverage(x), **rep,
                    bands={k: v for k, v in bands.items() if k != "samples"},
                ))
    return report


def run_propagation(
    cfg,
    src: str,
    dst: str,
    M_in: "gm.GlobalManifold",
    X_out: torch.Tensor,
    tr_i: torch.Tensor,
    va_i: torch.Tensor,
    latent_dim: int,
    seed: int,
    alphas: list[float],
    n_samples: int,
    epochs: int,
) -> dict:
    """Propagate source-manifold samples through the real ONNX src -> dst
    sublayer and validate against an independent destination fit. Only valid
    for tokenwise sublayer pairs declared in cfg.sublayers -- see
    gmanifold/onnx_transformer.py's make_map docstring."""
    Xtr, Xva = X_out[tr_i].to(DEVICE), X_out[va_i].to(DEVICE)
    M_out = gm.fit(gm.GlobalManifold(latent_dim=latent_dim, device=DEVICE), Xtr, epochs=epochs, X_val=Xva, seed=seed)
    T_out = gm.TangentCharts(Xtr, latent_dim)
    k_out = gm.KernelScore(Xtr)
    map_fn = tr.make_map(cfg, src, dst)

    bands = {k: v for k, v in gm.support_report(M_out, Xva, Xva, k_out, T_out).items() if k != "samples"}
    rows = {}
    for alpha in alphas:
        x, _ = M_in.sample(n_samples, alpha=alpha, auto_tau=True, seed=seed)
        rows[f"sample alpha={alpha}"] = gm.support_report(M_out, map_fn(x), kernel=k_out, tangent=T_out)["samples"]
    rows["real held-out images"] = gm.support_report(M_out, map_fn(M_in.states[va_i]), kernel=k_out, tangent=T_out)["samples"]
    return dict(src=src, dst=dst, rows=rows, bands=bands)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="Path to a toml ModelConfig")
    ap.add_argument("--out-dir", required=True, help="One report json is written here per location")
    ap.add_argument("--locs", nargs="+", required=True, help="Sublayer tensor names to fit a manifold at (e.g. embedding add_6 add_8)")
    ap.add_argument("--latent-dims", type=int, nargs="+", default=[16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--embedding-loc", default="embedding", help="Location name to treat as the vocab-embedding location")
    ap.add_argument("--propagate", nargs=2, metavar=("SRC", "DST"), action="append", default=[],
                     help="Also propagate the SRC location's samples through the real ONNX SRC->DST sublayer "
                          "(must be a declared tokenwise sublayer pair, e.g. --propagate add_6 add_8). Repeatable.")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute a location even if its report already exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    T0 = time.time()

    if args.embedding_loc not in args.locs:
        raise ValueError(
            f"--locs must include --embedding-loc ({args.embedding_loc!r}) -- it anchors the shared "
            "vocab/token-id universe and train/val split every other location's states are read at",
        )

    states = {}
    for loc in args.locs:
        if loc == args.embedding_loc:
            X_emb, ids = tr.vocab_states(cfg, embedding_output_name=loc)
            keep = ~gm.degenerate_mask(X_emb)
            n_dropped = int((~keep).sum())
            X_emb, ids = X_emb[keep], [i for i, k in zip(ids, keep.tolist()) if k]
            print(f"[{loc}] dropped {n_dropped} degenerate tokens; {len(ids)} remain", flush=True)
            states[loc] = (X_emb, ids)

    if states:
        # shared token/id set and train/val split across every embedding-derived location
        _, ids = next(iter(states.values()))
        non_embed_locs = [l for l in args.locs if l != args.embedding_loc]
        if non_embed_locs:
            input_ids = torch.zeros(len(ids), cfg.seq, dtype=torch.long)
            input_ids[:, 0] = torch.tensor(ids)
            positions = torch.zeros(len(ids), dtype=torch.long)
            collected = tr.collect_states(cfg, input_ids, positions, non_embed_locs)
            for loc in non_embed_locs:
                states[loc] = (collected[loc], ids)
        perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
        n_val = max(len(ids) // 10, 1)
        tr_i, va_i = perm[:-n_val], perm[-n_val:]
    else:
        tr_i = va_i = None

    fitted: dict[str, "gm.GlobalManifold"] = {}
    for loc in args.locs:
        path = _report_path(args.out_dir, loc)
        if os.path.exists(path) and not args.force:
            print(f"[{time.time() - T0:6.0f}s] {loc}: report already exists at {path}, skipping", flush=True)
            continue
        X, _ = states[loc]
        report = run_location(
            cfg, loc, X, tr_i, va_i,
            latent_dims=args.latent_dims, seeds=args.seeds, alphas=args.alphas,
            n_samples=args.n, epochs=args.epochs,
        )
        _write_report(args.out_dir, loc, report, args.no_push)
        print(f"[{time.time() - T0:6.0f}s] {loc}: report saved -> {path}", flush=True)

    for src, dst in args.propagate:
        if src not in states:
            raise ValueError(f"--propagate source {src!r} must be one of --locs {args.locs}")
        prop_key = f"{src}.{dst}"
        path = _report_path(args.out_dir, prop_key)
        if os.path.exists(path) and not args.force:
            print(f"[{time.time() - T0:6.0f}s] {prop_key}: report already exists at {path}, skipping", flush=True)
            continue
        X_in, ids_in = states[src]
        Xtr_in = X_in[tr_i].to(DEVICE)
        M_in = gm.fit(
            gm.GlobalManifold(latent_dim=args.latent_dims[0], device=DEVICE),
            Xtr_in, epochs=args.epochs, X_val=X_in[va_i].to(DEVICE), seed=args.seeds[0],
        )

        input_ids = torch.zeros(len(ids_in), cfg.seq, dtype=torch.long)
        input_ids[:, 0] = torch.tensor(ids_in)
        positions = torch.zeros(len(ids_in), dtype=torch.long)
        X_out = tr.collect_states(cfg, input_ids, positions, [dst])[dst]

        report = run_propagation(
            cfg, src, dst, M_in, X_out, tr_i, va_i,
            latent_dim=args.latent_dims[0], seed=args.seeds[0], alphas=[0.25, 0.5, 1.0],
            n_samples=args.n, epochs=args.epochs,
        )
        _write_report(args.out_dir, prop_key, report, args.no_push)
        print(f"[{time.time() - T0:6.0f}s] {prop_key}: propagation report saved -> {path}", flush=True)

    print(f"done in {time.time() - T0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
