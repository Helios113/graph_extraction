"""Plotly figures for series 1 (HTML + PNG each). Every panel plots a measured discrepancy (ORT pipeline vs
torch) against its pass criterion: confirmed when every mark lies below the red line."""
import json
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("BROWSER_PATH", "/nfs-share/pa511/tools/chrome/chrome-linux64/chrome")  # Chrome on NFS (home is per-node); needed by kaleido for PNG
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "series1"
EPS32 = float(np.finfo(np.float32).eps)
TOL = {"static": 1e-5, "traj": 1e-3}
FLOOR = 1e-10
FONT = dict(family="Times New Roman, DejaVu Serif, serif", size=12, color="#0b0b0b")
RED = "#d03b3b"

LOSS = {  # key: (label, colour, marker symbol)
    "JacobianExtractorLoss": ("JacobianExtractor", "#2a78d6", "circle"),
    "AdversarialLoss_SquaredDiffLoss": ("SquaredDiff", "#eb6834", "square"),
    "AdversarialLoss_NormalizedSquaredDiffLoss": ("NormalizedSqDiff", "#1baf7a", "triangle-up"),
    "AdversarialLoss_EnvelopeDiffLoss": ("EnvelopeDiff", "#6250d6", "diamond"),
}


def criterion(fig, row, col, tol, text="pass criterion", eps=True, ncols=2):
    n = (row - 1) * ncols + col
    ax = "" if n == 1 else str(n)
    fig.add_hrect(y0=tol, y1=tol * 1e4, fillcolor=RED, opacity=0.08, line_width=0, row=row, col=col)
    fig.add_hline(y=tol, line_color=RED, line_width=1.2, row=row, col=col)
    fig.add_annotation(x=0.01, y=np.log10(tol) + 0.1, xref=f"x{ax} domain", yref=f"y{ax}", text=f"{text} {tol:.0e} (fail above)",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=10, color="#a52a2a"))
    if eps and tol > EPS32:
        fig.add_hline(y=EPS32, line_color="#52514e", line_dash="dot", line_width=1, row=row, col=col)
        fig.add_annotation(x=0.99, y=np.log10(EPS32) - 0.1, xref=f"x{ax} domain", yref=f"y{ax}", text="float32 ε = 1.2e-7",
                           showarrow=False, xanchor="right", yanchor="top", font=dict(size=10, color="#52514e"))
    fig.update_yaxes(type="log", range=[np.log10(FLOOR), np.log10(tol * 1e2)], gridcolor="#e4e3df", exponentformat="power",
                     dtick=1, row=row, col=col)
    fig.update_xaxes(gridcolor="#e4e3df", row=row, col=col)


def traj_metric(rep):
    """max over steps/losses/optimisers of |L_ORT - L_torch| / max|L_torch| (scale of that trajectory)."""
    return max(float((np.abs(np.array(d["ort"]) - np.array(d["torch"])) / (np.abs(d["torch"]).max() + 1e-12)).max())
               for d in rep["optimisation"].values())


def save(fig, path_no_ext):
    fig.write_html(f"{path_no_ext}.html", include_plotlyjs="cdn")
    fig.write_image(f"{path_no_ext}.png", scale=2)


def plot_model(rd, rep):
    spans = list(rep["extraction"])
    spans += [t for t in {k.split("|")[-1] for k in rep["pullback"]} if t not in spans]
    lab = [s.replace("->", "→") for s in spans]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.17, horizontal_spacing=0.1, subplot_titles=(
        "(a) Extracted subgraph, forward", "(b) Pullback loss value",
        "(c) Gradient ∂L/∂input; ✕ = JacobianExtractor full Jacobian", "(d) Optimisation: SGD solid, AdamW dashed"))
    fig.add_trace(go.Scatter(x=lab, y=[max(rep["extraction"][s], FLOOR) for s in spans], mode="markers", name="forward",
                             marker=dict(color="#2a78d6", size=9, line=dict(color="white", width=1)), showlegend=False,
                             hovertemplate="%{x}: %{y:.2e}<extra>forward</extra>"), row=1, col=1)
    for (r, c, key) in [(1, 2, "loss_rel_err"), (2, 1, "grad_rel_err")]:
        for lk, (ll, col, sym) in LOSS.items():
            fig.add_trace(go.Scatter(x=lab, y=[max(rep["pullback"][f"{lk}|{s}"][key], FLOOR) if f"{lk}|{s}" in rep["pullback"] else None for s in spans], mode="markers",
                                     name=ll, legendgroup=ll, showlegend=(key == "loss_rel_err"),
                                     marker=dict(color=col, symbol=sym, size=9, line=dict(color="white", width=1)),
                                     hovertemplate="%{x}: %{y:.2e}<extra>" + ll + "</extra>"), row=r, col=c)
    if rep.get("jacobian"):
        fig.add_trace(go.Scatter(x=lab, y=[max(rep["jacobian"][s], FLOOR) if s in rep["jacobian"] else None for s in spans], mode="markers",
                                 name="JacobianExtractor full Jacobian", showlegend=True,
                                 marker=dict(color="black", symbol="x", size=10),
                                 hovertemplate="%{x}: %{y:.2e}<extra>full Jacobian</extra>"), row=2, col=1)
    for lk, (ll, col, _) in LOSS.items():
        for opt, dash in [("SGD", "solid"), ("AdamW", "dash")]:
            if f"{lk}|{opt}" not in rep["optimisation"]:
                continue
            d = rep["optimisation"][f"{lk}|{opt}"]
            o, t = np.array(d["ort"]), np.array(d["torch"])
            fig.add_trace(go.Scatter(y=np.maximum(np.abs(o - t) / (np.abs(t).max() + 1e-12), FLOOR), mode="lines", name=f"{ll} {opt}",
                                     legendgroup=ll, showlegend=False, line=dict(color=col, dash=dash, width=1.4),
                                     hovertemplate="step %{x}: %{y:.2e}<extra>" + f"{ll} {opt}" + "</extra>"), row=2, col=2)
    for r, c in [(1, 1), (1, 2), (2, 1)]:
        criterion(fig, r, c, TOL["static"])
        fig.update_xaxes(title_text="subgraph (boundary tensors)", row=r, col=c)
    criterion(fig, 2, 2, TOL["traj"], eps=False)
    fig.update_xaxes(title_text="optimisation step", row=2, col=2)
    fig.update_yaxes(title_text=rep.get("extraction_label", "max |ORT − torch|"), row=1, col=1)
    fig.update_yaxes(title_text="relative error", row=1, col=2)
    fig.update_yaxes(title_text="‖Δ‖∞ / ‖g_torch‖∞", row=2, col=1)
    fig.update_yaxes(title_text="|L_ORT − L_torch| / max|L_torch|", row=2, col=2)
    fig.update_layout(
        width=1000, height=820, font=FONT, plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"{rep['model']}: ORT pipeline vs. torch autograd<br><sup>ops: {', '.join(rep['ops'])}. "
                        "Confirmed when every mark lies below the red criterion line; the dotted line is the best float32 can do.</sup>", x=0.01),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"), margin=dict(t=110, b=80))
    if rep.get("errors"):
        errs = {k: v for k, v in rep["errors"].items() if not k.startswith("note|")}
        kinds = {}
        for k, v in errs.items():
            kinds.setdefault(f"{k.split('|')[0].split(':')[0]}: {v[:60]}", set()).add(k.split("|")[-1])
        txt = "<b>Failed (no data plotted):</b><br>" + "<br>".join(f"{m} [{', '.join(sorted(sp))}]" for m, sp in list(kinds.items())[:4])
        fig.add_annotation(x=0.5, y=0.5, xref="x4 domain", yref="y4 domain", text=txt, showarrow=False, align="left", font=dict(size=10, color="#a52a2a"),
                           bgcolor="rgba(255,255,255,0.9)", bordercolor="#d03b3b", borderwidth=1) if not rep["optimisation"] else \
            fig.add_annotation(x=0.0, y=-0.2, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, align="left", text=txt, font=dict(size=9, color="#a52a2a"))
    for a in fig.layout.annotations[:4]:  # subplot titles: left-aligned over the axes
        a.update(xanchor="left", x=fig.layout[f"xaxis{'' if a.text.startswith('(a)') else {'(b)': 2, '(c)': 3, '(d)': 4}[a.text[:3]]}"].domain[0], font=dict(size=13))
    save(fig, rd / "results")


def plot_summary(reps):
    names = list(reps)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.1,
                        subplot_titles=("(a) Static checks", "(b) Trajectory"))
    for k, ll, col, sym in [("extract_max", "forward", "#2a78d6", "circle"), ("loss_rel_max", "loss value", "#eb6834", "square"),
                            ("grad_rel_max", "gradient", "#1baf7a", "triangle-up")]:
        fig.add_trace(go.Scatter(x=names, y=[reps[n][k] for n in names], mode="markers", name=ll,
                                 marker=dict(color=col, symbol=sym, size=9, line=dict(color="white", width=1))), row=1, col=1)
    fig.add_trace(go.Scatter(x=names, y=[reps[n]["opt_traj_max"] for n in names], mode="markers", name="trajectory",
                             marker=dict(color="#6250d6", symbol="diamond", size=9, line=dict(color="white", width=1))), row=1, col=2)
    criterion(fig, 1, 1, TOL["static"])
    criterion(fig, 1, 2, TOL["traj"], text="criterion", eps=False)
    fig.update_yaxes(title_text="worst case over spans and losses", row=1, col=1)
    fig.update_yaxes(title_text="max |ΔL| / max|L|", row=1, col=2)
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(width=1000, height=470, font=FONT, plot_bgcolor="white", paper_bgcolor="white",
                      title=dict(text="Series 1: worst-case ORT-vs-torch discrepancy per network", x=0.01),
                      legend=dict(orientation="h", y=-0.3, x=0.35, xanchor="center"), margin=dict(t=80, b=110))
    save(fig, RESULTS / "summary")


def rebuild(results_dir):
    """Re-plot every <results_dir>/*/results.json and rebuild summary.json + the summary figure."""
    global RESULTS
    RESULTS = Path(results_dir)
    summary = {}
    for f in sorted(RESULTS.glob("*/results.json")):
        rep = json.load(open(f))
        plot_model(f.parent, rep)
        if rep["extraction"] and rep["pullback"] and rep["optimisation"]:
            summary[rep["model"]] = {
                "extract_max": max(rep["extraction"].values()),
                "grad_rel_max": max(v["grad_rel_err"] for v in rep["pullback"].values()),
                "loss_rel_max": max(v["loss_rel_err"] for v in rep["pullback"].values()),
                "opt_traj_max": traj_metric(rep),
                "jac_rel_max": max(rep.get("jacobian", {}).values(), default=None),
            }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    if summary:
        plot_summary(summary)


if __name__ == "__main__":
    import sys
    rebuild(sys.argv[1] if len(sys.argv) > 1 else RESULTS)
