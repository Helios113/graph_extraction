"""Plots for series 4 (KV optimisation trajectories, test outcomes) and series 5 (time / memory). plotly -> html + png."""
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ.setdefault("BROWSER_PATH", "/nfs-share/pa511/tools/chrome/chrome-linux64/chrome")  # Chrome on NFS, needed by kaleido for PNG
ROOT = Path(__file__).resolve().parent.parent
S4, S5 = ROOT / "results" / "series4", ROOT / "results" / "series5"


def save(fig, d, name):
    fig.write_html(d / f"{name}.html")
    fig.write_image(d / f"{name}.png", scale=2)


# ---- series 4: test outcomes per file
cnt = Counter()
for tc in ET.parse(S4 / "junit.xml").getroot().iter("testcase"):
    cnt[(tc.get("classname").split(".")[-1].removeprefix("test_kv_"), "fail" if tc.find("failure") is not None else "pass")] += 1
files = sorted({k[0] for k in cnt})
fig = go.Figure([go.Bar(name=s, x=files, y=[cnt[(f, s)] for f in files], marker_color=c) for s, c in (("pass", "#2a9d8f"), ("fail", "#e63946"))])
fig.update_layout(barmode="stack", title="Series 4: KV tests per file (121 total)", yaxis_title="tests", xaxis_tickangle=-30, template="plotly_white", height=550, width=1000)
save(fig, S4, "test_outcomes")

# ---- series 4: ORT vs torch trajectory diff, and example trajectories
opt = json.load(open(S4 / "optimisation.json"))
fig = go.Figure()
for o, col in (("SGD", "#1d3557"), ("AdamW", "#e76f51")):
    ks = [k for k in opt if k.split("|")[2] == o]
    fig.add_trace(go.Scatter(x=ks, y=[opt[k]["traj_diff"] for k in ks], mode="markers", name=o, marker=dict(color=col, size=8)))
fig.add_hline(y=1e-3, line_dash="dash", annotation_text="tolerance 1e-3")
fig.update_layout(title="Series 4: max relative trajectory difference, ORT KV pullback vs torch autograd (100-step runs)",
                  yaxis_type="log", yaxis_title="rel. diff", xaxis_tickangle=-60, template="plotly_white", height=800, width=1500, margin=dict(b=380))
save(fig, S4, "trajectory_diff")

fig = make_subplots(rows=1, cols=2, subplot_titles=("attention L8, SquaredDiff", "mlp L0, SquaredDiff (SGD diverges in both)"))
for col, (kind, L) in enumerate((("attention", 8), ("mlp", 0)), 1):
    for o, c in (("SGD", "#1d3557"), ("AdamW", "#e76f51")):
        d = opt[f"{kind}|L{L}|{o}|AdversarialLoss_SquaredDiffLoss"]
        fig.add_trace(go.Scatter(y=d["ort"], name=f"{o} ORT", line=dict(color=c), legendgroup=o, showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(y=d["torch"], name=f"{o} torch", line=dict(color=c, dash="dot"), legendgroup=o, showlegend=col == 1), row=1, col=col)
fig.update_yaxes(type="log")
fig.update_layout(title="Series 4: loss trajectories, ORT (solid) vs torch (dotted)", xaxis_title="step", xaxis2_title="step", template="plotly_white", height=500, width=1200)
save(fig, S4, "trajectories")

# ---- series 4: KV timing (written by test_kv_timing)
tm = json.load(open(ROOT / "tests" / "kv_tests" / "results" / "timing.json"))
print("timing.json keys:", list(tm)[:6] if isinstance(tm, dict) else type(tm))

# ---- series 5
r = json.load(open(S5 / "results.json"))["rows"]
cols = {"kv_onnx": "#1d3557", "cacheless_onnx": "#e63946", "torch_cache": "#2a9d8f", "torch": "#f4a261"}
for key, title, ylab, logy, name in (("ms", "time per 768-candidate group", "ms (median)", True, "time"),
                                     ("peak_mb", "peak device memory (above CUDA context)", "MB", False, "memory")):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("forward, boundary 24", "pullback (768x768 Jacobian, attn L1)"))
    for ci, op in enumerate(("fwd", "pb"), 1):
        for v, c in cols.items():
            rows = sorted((x for x in r if x["op"] == op and x["variant"] == v), key=lambda x: x["L"])
            fig.add_trace(go.Scatter(x=[x["L"] for x in rows], y=[x[key] for x in rows], mode="lines+markers", name=v,
                                     line=dict(color=c), legendgroup=v, showlegend=ci == 1), row=1, col=ci)
    fig.update_xaxes(type="log" if False else "linear", title_text="prefix length L")
    fig.update_yaxes(type="log" if logy else "linear", title_text=ylab)
    fig.update_layout(title=f"Series 5 (GPT-2, H100, fp32): {title}", template="plotly_white", height=500, width=1200)
    save(fig, S5, name)
