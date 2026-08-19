"""Config-driven optimization over a non-weight input.

Each sublayer's target is optimized against its own AdversarialLoss_* gradient
graph (loss_type is read from the config's toml -- unlike Jacobian discovery,
which always forces MaskedSumLoss, see execute.find_jacobians). The input being
optimized (SublayerConfig.input) is a plain runtime input, never a graph
initializer, so it is never picked up by onnxblock's SGD/AdamW optimizer build;
execute.optimization_loop.optimization_loop instead runs a manual numpy update
loop driven by the graph's own d(loss)/d(input) output, using a pluggable
Optimizer (see execute.optimization_loop.SGD/AdamW/NewtonRaphson).
"""

import argparse
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import onnxruntime as ort

from config import LossType, ModelConfig, load_config
from construct.construct import ensure_base_model, ensure_subgraph_pullback, ensure_subgraphs
from execute.optimization_loop import AdamW, NewtonRaphson, Optimizer, SGD, optimization_loop


def _random_input(shape: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _forward_target(subgraph_path: Path, input_name: str, output_name: str, x: np.ndarray) -> np.ndarray:
    """Computes target = f(x) with a plain forward-only session on the extracted
    subgraph, before any gradient/loss nodes are added."""
    session = ort.InferenceSession(str(subgraph_path))
    return session.run([output_name], {input_name: x})[0]


def run_optimization(
    cfg: ModelConfig,
    optimizer_factory: Callable[[], Optimizer],
    steps: int = 500,
    init_perturbation: float = 1e-2,
    seed: int = 0,
    force: bool = False,
) -> dict[str, tuple[np.ndarray, list[float]]]:
    """Runs one optimization loop per sublayer pair in cfg.sublayers, each against
    its own configured AdversarialLoss_* graph. Returns
    {f"{input}.{output}": (y_final, loss_history)}.

    optimizer_factory: called once per sublayer to build a fresh Optimizer -- a
    stateful optimizer (AdamW's moment buffers, NewtonRaphson's previous
    y/grad) is shaped to one y, so it can't be shared across sublayers of
    different shapes; pass e.g. `lambda: SGD(lr=0.1)`, `lambda: AdamW(lr=1e-3)`,
    or `lambda: NewtonRaphson(lr=0.1)`.
    """
    ensure_base_model(cfg, force=force)
    sub_graphs = ensure_subgraphs(cfg, force=force)

    results: dict[str, tuple[np.ndarray, list[float]]] = {}

    for path, sublayer_conf in sub_graphs:
        if sublayer_conf.loss_type == LossType.MaskedSumLoss:
            raise ValueError(
                f"{cfg.name}: sublayer {sublayer_conf.input}->{sublayer_conf.output} is "
                "configured with MaskedSumLoss, which is only used for Jacobian discovery "
                "(see execute.find_jacobians) -- set loss_type to one of the "
                "AdversarialLoss_* variants to run the optimization loop",
            )

        generated = ensure_subgraph_pullback([(path, sublayer_conf)], force=force)
        gradient_path = generated["gradient"][0]
        gradient_session = ort.InferenceSession(str(gradient_path))

        x = _random_input(sublayer_conf.input_shape, seed)
        target = _forward_target(path, sublayer_conf.input, sublayer_conf.output, x)
        y0 = x + init_perturbation * _random_input(sublayer_conf.input_shape, seed + 1)

        session_input_names = {i.name for i in gradient_session.get_inputs()}
        needs_anchor = "x" in session_input_names and sublayer_conf.input != "x"

        y_final, loss_history = optimization_loop(
            gradient_session,
            y0=y0,
            target=target,
            input_name=sublayer_conf.input,
            steps=steps,
            optimizer=optimizer_factory(),
            x=x if needs_anchor else None,
        )
        results[f"{sublayer_conf.input}.{sublayer_conf.output}"] = (y_final, loss_history)

    return results


def save_results(results: dict[str, tuple[np.ndarray, list[float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, (y_final, loss_history) in results.items():
            group = f.create_group(name)
            group.create_dataset("y_final", data=y_final)
            group.create_dataset("loss_history", data=np.asarray(loss_history))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("--force", action="store_true", help="Rebuild graphs even if cached")
    parser.add_argument("--steps", type=int, default=500, help="Update steps")
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw", "newton"],
        default="sgd",
        help="Update rule to use (see execute.optimization_loop for the full Optimizer interface)",
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument(
        "--init-perturbation",
        type=float,
        default=1e-2,
        help="y0 = x + this much random perturbation",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for x and the y0 perturbation")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to save the .h5 results (default: <config.h5_dir>/optimization.h5)",
    )
    args = parser.parse_args()

    optimizer_factories: dict[str, Callable[[], Optimizer]] = {
        "sgd": lambda: SGD(lr=args.lr),
        "adamw": lambda: AdamW(lr=args.lr),
        "newton": lambda: NewtonRaphson(lr=args.lr),
    }

    cfg = load_config(args.config)
    results = run_optimization(
        cfg,
        optimizer_factory=optimizer_factories[args.optimizer],
        steps=args.steps,
        init_perturbation=args.init_perturbation,
        seed=args.seed,
        force=args.force,
    )

    out_path = Path(args.out) if args.out else cfg.h5_dir / "optimization.h5"
    save_results(results, out_path)

    for name, (y_final, loss_history) in results.items():
        print(f"{name}: final loss {loss_history[-1]:.6g} over {len(loss_history)} steps")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
