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
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(subgraph_path), sess_options=session_options)
    return session.run([output_name], {input_name: x})[0]


def run_optimization(
    cfg: ModelConfig,
    optimizer_factory: Callable[[], Optimizer],
    steps: int = 500,
    init_perturbation: float = 1e-2,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, list[float]]]:

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
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        gradient_session = ort.InferenceSession(str(gradient_path), sess_options=session_options)

        x = _random_input(sublayer_conf.input_shape, seed)
        target = _forward_target(path, sublayer_conf.input, sublayer_conf.output, x)
        y0 = x + init_perturbation * _random_input(sublayer_conf.input_shape, seed + 1)

        session_input_names = {i.name for i in gradient_session.get_inputs()}
        needs_anchor = "x" in session_input_names and sublayer_conf.input != "x"



    return results


def save_results(results: dict[str, tuple[np.ndarray, list[float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, (y_final, loss_history) in results.items():
            group = f.create_group(name)
            group.create_dataset("y_final", data=y_final)
            group.create_dataset("loss_history", data=np.asarray(loss_history))


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
#     parser.add_argument("--force", action="store_true", help="Rebuild graphs even if cached")
#     parser.add_argument("--steps", type=int, default=500, help="Update steps")
#     parser.add_argument(
#         "--optimizer",
#         choices=["sgd", "adamw", "newton"],
#         default="sgd",
#         help="Update rule to use (see execute.optimization_loop for the full Optimizer interface)",
#     )
#     parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
#     parser.add_argument(
#         "--init-perturbation",
#         type=float,
#         default=1e-2,
#         help="y0 = x + this much random perturbation",
#     )
#     parser.add_argument("--seed", type=int, default=0, help="RNG seed for x and the y0 perturbation")
#     parser.add_argument(
#         "--out",
#         type=str,
#         default=None,
#         help="Where to save the .h5 results (default: <config.h5_dir>/optimization.h5)",
#     )
#     args = parser.parse_args()

#     optimizer_factories: dict[str, Callable[[], Optimizer]] = {
#         "sgd": lambda: SGD(lr=args.lr),
#         "adamw": lambda: AdamW(lr=args.lr),
#         "newton": lambda: NewtonRaphson(lr=args.lr),
#     }

#     cfg = load_config(args.config)
#     results = run_optimization(
#         cfg,
#         optimizer_factory=optimizer_factories[args.optimizer],
#         steps=args.steps,
#         init_perturbation=args.init_perturbation,
#         seed=args.seed,
#         force=args.force,
#     )

#     out_path = Path(args.out) if args.out else cfg.h5_dir / "optimization.h5"
#     save_results(results, out_path)

#     for name, (y_final, loss_history) in results.items():
#         print(f"{name}: final loss {loss_history[-1]:.6g} over {len(loss_history)} steps")
#     print(f"Saved -> {out_path}")


# if __name__ == "__main__":
#     main()
