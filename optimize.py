"""Gradient-based collision search: for each sublayer, samples a fixed point x and a nearby
trainable point y0, then runs cfg.collision_opt_steps of Adam/SGD on y (x held fixed) to
minimize ||f(x) - f(y)||^2, looking for a distinct point that collides with x's output.

Usage: from collision_opt.optimize import run_collision_opt
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ModelConfig
from optimise_activations import sample_x, sample_ball
from build_twin_graph import (
    checkpoint_path, eval_model_path, optimizer_model_path, training_model_path, y_input_name,
)
from post_utils import save_collision_opt_results


def optimize_collision(cfg: ModelConfig, sublayer_idx: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Gradient-descends y (x held fixed) towards a collision with f(x), for cfg.batch
    independent (x, y) restarts at once. x is sampled once via sample_x; y0 is x perturbed by
    collision_opt_init_perturbation (via sample_ball). Both are tiled across every seq slot
    (see max_search.py's g() closure) before being written into the twin graph, which is
    itself batch/seq/d_model-shaped."""
    sublayer = cfg.sublayers[sublayer_idx]
    position = cfg.collision_opt_position

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    state = CheckpointState.load_checkpoint(checkpoint_path(cfg, sublayer_idx))
    model = Module(
        training_model_path(cfg, sublayer_idx), state, eval_model_path(cfg, sublayer_idx),
        device="cuda", session_options=session_options,
    )
    optimizer = Optimizer(optimizer_model_path(cfg, sublayer_idx), model)
    optimizer.set_learning_rate(cfg.collision_opt_lr)

    x = sample_x(cfg, sublayer_idx, position, seed)
    y0 = sample_ball(x, cfg.collision_opt_init_perturbation, cfg.batch, seed=seed + 1)
    state.parameters[y_input_name(sublayer)].data = np.tile(y0[:, None, :], (1, cfg.seq, 1)).astype(np.float32)

    x_tiled = np.tile(x[:, None, :], (1, cfg.seq, 1)).astype(np.float32)
    print(x_tiled.shape)
    print(y0.shape)
    model.train()
    losses = np.empty(cfg.collision_opt_steps, dtype=np.float32)
    for step in tqdm(range(cfg.collision_opt_steps), desc=f"sublayer{sublayer_idx} collision_opt"):
        loss = model(x_tiled)
        losses[step] = loss
        print(loss)
        optimizer.step()
        model.lazy_reset_grad()
    
    y_final = state.parameters[y_input_name(sublayer)].data[:, position, :]
    domain_distance_final = np.linalg.norm(y_final - x, axis=-1)
    print(y_final.shape)
    return {
        "x": x,
        "y_final": y_final,
        "loss_curve": losses,
        "loss_final": losses[-1:].repeat(cfg.batch),
        "domain_distance_final": domain_distance_final,
    }



def run_collision_opt(cfg: ModelConfig) -> None:
    all_results = {}
    for sublayer_idx in range(len(cfg.sublayers)):
        results = optimize_collision(cfg, sublayer_idx, seed=cfg.seed)
        mean_loss = float(np.mean(results["loss_final"]))
        mean_dist = float(np.mean(results["domain_distance_final"]))
        print(f"sublayer{sublayer_idx}: mean loss = {mean_loss:.6g}, mean ||x-y|| = {mean_dist:.6g}")
        all_results[sublayer_idx] = results
    save_collision_opt_results(cfg, all_results)
