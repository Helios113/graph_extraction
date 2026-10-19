"""Series 2: the series-1 pipeline on networks WITH control flow (ONNX If / Loop), exported through TorchScript.
Reuses tests/series1/run_series1.run_model (torch autograd on the un-scripted Python module is the baseline).
A failing extraction / pullback is a finding: it is recorded in results.json ("errors") and shown on the plot."""
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "series1"))
import run_series1 as S1  # noqa: E402
import series1_plots as P  # noqa: E402

D = S1.D


class CondStage(nn.Module):
    """Data-dependent branch on the whole batch: tanh(A x) if sum(x) > 0 else sigmoid(B x)."""
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        if x.sum() > 0:
            return torch.tanh(self.a(x))
        else:
            return torch.sigmoid(self.b(x))


class LoopStage(nn.Module):
    """Fixed trip-count loop: h <- tanh(W h) + 0.5 h, 4 times."""
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, D)

    def forward(self, x):
        h = x
        for _ in range(4):
            h = torch.tanh(self.w(h)) + 0.5 * h
        return h


class BreakLoopStage(nn.Module):
    """Data-dependent loop exit: up to 6 applications of tanh(W h), stopping once max|h| < 0.3."""
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, D)

    def forward(self, x):
        h = x
        for _ in range(6):
            h = torch.tanh(self.w(h))
            if h.abs().max() < 0.3:
                break
        return h


class NestedStage(nn.Module):
    """Loop whose body contains a branch (If inside Loop)."""
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        h = x
        for _ in range(3):
            if h.mean() > 0:
                h = torch.tanh(self.a(h))
            else:
                h = torch.tanh(self.b(h)) - 0.2
        return h


def lin():
    return nn.Sequential(nn.Linear(D, D), nn.Tanh())


MODELS = {  # name: (stages, shape, shift)
    "cond_branch_pos": ([lin(), CondStage(), lin()], (S1.B, D), 3.0),
    "cond_branch_neg": ([lin(), CondStage(), lin()], (S1.B, D), -3.0),
    "loop_fixed": ([lin(), LoopStage(), lin()], (S1.B, D), 0.0),
    "loop_break": ([lin(), BreakLoopStage(), lin()], (S1.B, D), 0.0),
    "loop_with_cond": ([lin(), NestedStage(), lin()], (S1.B, D), 0.0),
}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODELS):
        print("=== ", n, flush=True)
        st, shape, shift = MODELS[n]
        try:
            r = S1.run_model(n, st, shape, series="series2", allow_cf=True, script=True, shift=shift)
            print(n, "ops", r["ops"], "errors", len(r["errors"]), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"MODEL FAILED {n}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    P.rebuild(S1.ROOT / "results" / "series2")
