"""Plotly figures for series 4 (HTML + PNG each), in the style of series 1-3: every panel plots a measured
ORT-KV-vs-torch discrepancy against its pass criterion; confirmed when every mark lies below the red line."""
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "series1"))
import series1_plots as P1  # noqa: E402  (criterion, save, FONT, LOSS, FLOOR)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "series4"
TOL = {"forward": 1e-4, "jacobian": 1e-3, "traj": 1e-3}
LCOL = {0: "#2a78d6", 1: "#eb6834", 8: "#1baf7a", 32: "#6250d6"}  # prefix length -> colour
MARK = dict(size=9, line=dict(color="white", width=1))


def plot_model(rd, rep):
    Ls = rep["prefix_lengths"]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.17, horizontal_spacing=0.1, subplot_titles=(
        "(a) Last-token boundary activations, KV vs HF", "(b) Last-token logits, KV vs HF",
        "(c) Last-token Jacobian (full d×d), KV pullback vs torch", "(d) Optimisation: SGD filled, AdamW open"))
    for L in Ls:
        e = rep["forward"].get(str(L))
        if e:
            fig.add_trace(go.Scatter(x=list(range(len(e["boundaries"]))), y=[max(v, P1.FLOOR) for v in e["boundaries"]], mode="lines+markers",
                                     name=f"L = {L}", legendgroup=f"L{L}", line=dict(color=LCOL[L], width=1.2),
                                     marker=dict(color=LCOL[L], size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[f"L = {L}" for L in Ls if str(L) in rep["logits"]], y=[max(rep["logits"][str(L)], P1.FLOOR) for L in Ls if str(L) in rep["logits"]],
                             mode="markers", showlegend=False, marker=dict(color="#2a78d6", **MARK)), row=1, col=2)
    jt = sorted({k.split("|")[0] for k in rep["jacobian"]}, key=lambda s: int(s))
    for L in Ls:
        ys = [rep["jacobian"].get(f"{j}|{L}") for j in jt]
        fig.add_trace(go.Scatter(x=[f"h{j}→h{int(j) + 1}" for j in jt], y=[max(v["rel_fro"], P1.FLOOR) if v else None for v in ys], mode="markers",
                                 name=f"L = {L}", legendgroup=f"L{L}", showlegend=False, marker=dict(color=LCOL[L], **MARK)), row=2, col=1)
    cats = []
    for kind in ("attention", "mlp"):
        for L in Ls:
            if L:
                cats.append((kind, L))
    for lk, (ll, col, sym) in P1.LOSS.items():
        for opt in ("SGD", "AdamW"):
            xs, ys = [], []
            for kind, L in cats:
                d = rep["optimisation"].get(f"{kind}|L{L}|{opt}|{lk}")
                if d:
                    xs.append(f"{kind} L={L}"); ys.append(max(d["traj_diff"], P1.FLOOR))
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=ll, legendgroup=ll, showlegend=(opt == "SGD" and lk != "JacobianExtractorLoss"),
                                         marker=dict(color=col if opt == "SGD" else "white", symbol=sym, size=9, line=dict(color=col, width=1.6))), row=2, col=2)
    for r, c, tol in [(1, 1, TOL["forward"]), (1, 2, TOL["forward"]), (2, 1, TOL["jacobian"])]:
        P1.criterion(fig, r, c, tol)
    P1.criterion(fig, 2, 2, TOL["traj"], eps=False)
    fig.update_xaxes(title_text="residual boundary j (0 = embedding)", row=1, col=1)
    fig.update_xaxes(title_text="sublayer", row=2, col=1)
    fig.update_xaxes(title_text="sublayer type, prefix length", row=2, col=2)
    fig.update_yaxes(title_text="max |KV − HF| / max |HF|", row=1, col=1)
    fig.update_yaxes(title_text="max |KV − HF| / max |HF|", row=1, col=2)
    fig.update_yaxes(title_text="‖J_KV − J_torch‖_F / ‖J_torch‖_F", row=2, col=1)
    fig.update_yaxes(title_text="max |L_ORT − L_torch| / max |L_torch|", row=2, col=2)
    fig.update_layout(
        width=1050, height=840, font=P1.FONT, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"{rep['model']} (KV cache): ORT pipeline vs. torch<br><sup>{rep['n_layers']} layers, {rep['n_candidates']} candidate last tokens per shared prefix, "
                        "ORT CUDA fp32 (TF32 off). Confirmed when every mark lies below the red criterion line.</sup>", x=0.01),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), margin=dict(t=110, b=80))
    if rep.get("errors"):
        txt = "<b>Failed (no data plotted):</b><br>" + "<br>".join(f"{k}: {v[:70]}" for k, v in list(rep["errors"].items())[:5])
        fig.add_annotation(x=0.0, y=-0.2, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, align="left",
                           text=txt, font=dict(size=9, color="#a52a2a"))
    for i, a in enumerate(fig.layout.annotations[:4]):  # subplot titles: left-aligned over the axes
        a.update(xanchor="left", x=fig.layout[f"xaxis{'' if i == 0 else i + 1}"].domain[0], font=dict(size=13))
    P1.save(fig, rd / "results")


def summarise(rep):
    f = [max(e["boundaries"]) for e in rep["forward"].values()]
    return {"forward_max": max(f, default=None), "logits_max": max(rep["logits"].values(), default=None),
            "jacobian_max": max((v["rel_fro"] for v in rep["jacobian"].values()), default=None),
            "traj_max": max((v["traj_diff"] for v in rep["optimisation"].values()), default=None),
            "errors": len(rep["errors"])}


def plot_summary(summary):
    names = list(summary)
    fig = make_subplots(rows=1, cols=3, column_widths=[0.4, 0.3, 0.3], horizontal_spacing=0.09,
                        subplot_titles=("(a) Forward", "(b) Jacobian", "(c) Optimisation trajectory"))
    for k, ll, col, sym, c in [("forward_max", "boundaries", "#2a78d6", "circle", 1), ("logits_max", "logits", "#eb6834", "square", 1),
                               ("jacobian_max", "Jacobian", "#1baf7a", "triangle-up", 2), ("traj_max", "trajectory", "#6250d6", "diamond", 3)]:
        fig.add_trace(go.Scatter(x=names, y=[summary[n][k] for n in names], mode="markers", name=ll,
                                 marker=dict(color=col, symbol=sym, **MARK)), row=1, col=c)
    P1.criterion(fig, 1, 1, TOL["forward"], ncols=3)
    P1.criterion(fig, 1, 2, TOL["jacobian"], text="criterion", eps=False, ncols=3)
    P1.criterion(fig, 1, 3, TOL["traj"], text="criterion", eps=False, ncols=3)
    fig.update_yaxes(title_text="worst case over prefix lengths / sublayers / losses", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(width=1150, height=480, font=P1.FONT, plot_bgcolor="white", paper_bgcolor="white",
                      title=dict(text="Series 4: worst-case KV-cache ORT-vs-torch discrepancy per model", x=0.01),
                      legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"), margin=dict(t=80, b=110))
    for i, a in enumerate(fig.layout.annotations[:3]):
        a.update(xanchor="left", x=fig.layout[f"xaxis{'' if i == 0 else i + 1}"].domain[0], font=dict(size=13))
    P1.save(fig, RESULTS / "summary")


def rebuild(results_dir=RESULTS):
    """Re-plot every <results_dir>/*/results.json and rebuild summary.json + the summary figure."""
    global RESULTS
    RESULTS = Path(results_dir)
    summary = {}
    for f in sorted(RESULTS.glob("*/results.json")):
        rep = json.load(open(f))
        plot_model(f.parent, rep)
        summary[rep["model"]] = summarise(rep)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    if summary:
        plot_summary(summary)


if __name__ == "__main__":
    rebuild(sys.argv[1] if len(sys.argv) > 1 else RESULTS)
