"""gmanifold: learn G: R^m -> R^D from real states and sample ~uniformly on the learned manifold near the support.

Migrated from the standalone gmanifold repo. The manifold/geometry/tangent/validate
modules are unchanged pure PyTorch; `onnx_transformer` replaces the original
HF-forward-hook `transformer` module with this repo's ONNX pipeline
(construct.construct + onnxruntime), per CLAUDE.md's sublayer/config conventions.
"""
from .concurrent import fit_concurrent
from .geometry import degenerate_mask, intrinsic_dimension, knn, outlier_mask
from .manifold import GlobalManifold, fit
from .persistence import reattach
from .tangent import TangentCharts
from .validate import KernelScore, check_sampling, support_report

__all__ = ["GlobalManifold", "fit", "fit_concurrent", "TangentCharts", "KernelScore", "support_report", "check_sampling", "intrinsic_dimension", "knn", "degenerate_mask", "outlier_mask", "reattach"]
