import sys, tempfile, numpy as np, onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
import onnxruntime as ort
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from construct.pullback import generate_subgraph_pullback
from config import LossType

B = int(sys.argv[1]); lt = LossType[sys.argv[2]]; D = 16
d = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
rng = np.random.default_rng(0)
W = rng.normal(size=(D, D)).astype(np.float32)
x_i = h.make_tensor_value_info("hidden_in", T.FLOAT, [B, 1, D])
g = h.make_graph([h.make_node("MatMul", ["hidden_in", "W"], ["hidden_out"])], "g", [x_i],
                 [h.make_tensor_value_info("hidden_out", T.FLOAT, [B, 1, D])], [nh.from_array(W, "W")])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 15)]); m.ir_version = 8
sub = d / "sub.onnx"; onnx.save(m, sub)
pb = generate_subgraph_pullback(sub, lt, [B, 1, D], "hidden_in", "hidden_out")
xin = rng.normal(size=(B, 1, D)).astype(np.float32)
y = rng.normal(size=(B, 1, D)).astype(np.float32)
fx = xin @ W
feeds = {"x": xin, "hidden_in": y, "target": fx}
res = {}
for prov in ["CPUExecutionProvider", "CUDAExecutionProvider"]:
    s = ort.InferenceSession(str(pb), providers=[prov])
    ins = {i.name for i in s.get_inputs()}; outs = [o.name for o in s.get_outputs()]
    r = dict(zip(outs, s.run(outs, {k: v for k, v in feeds.items() if k in ins})))
    res[prov] = r
c, g_ = res["CPUExecutionProvider"], res["CUDAExecutionProvider"]
# finite diff on CPU loss
for k in c:
    print("RESULT", lt.name, B, k, "shape", c[k].shape, "cpu-vs-cuda maxdiff", float(np.abs(c[k] - g_[k]).max()), "cpu absmax", float(np.abs(c[k]).max()))

gc, gg = c["hidden_in_grad"].reshape(B, -1), g_["hidden_in_grad"].reshape(B, -1)
print("RESULT rows_zero_cuda", [int(i) for i in np.where(np.abs(gg).max(1) == 0)[0]][:20], "rows_bad", [int(i) for i in np.where(np.abs(gg-gc).max(1) > 1e-4)[0]][:20])
mm = onnx.load(str(pb), load_external_data=False)
for n in mm.graph.node:
    print("NODE", n.op_type, n.name, list(n.input), "->", list(n.output), n.domain)
for i in mm.graph.input: print("INPUT", i.name, [x.dim_value or x.dim_param for x in i.type.tensor_type.shape.dim])
for i in mm.graph.initializer: print("INIT", i.name, list(i.dims))
for o in mm.graph.output: print("OUTPUT", o.name, [x.dim_value or x.dim_param for x in o.type.tensor_type.shape.dim])
for v in mm.graph.value_info:
    if "div_output" in v.name or "reducesum_output" in v.name: print("VI", v.name, [x.dim_value or x.dim_param for x in v.type.tensor_type.shape.dim])
# hypothesis: seed `div_output_grad` is shape [1] but DivGrad's A/B are [B,1]; expand seed to full shape
full = onnx.load(str(pb))
for k, init in enumerate(full.graph.initializer):
    if init.name == "onnx::div_output::17_grad":
        full.graph.initializer[k].CopyFrom(nh.from_array(np.ones((B, 1), np.float32), init.name))
p2 = d / "fixed.onnx"; onnx.save(full, p2)
s = ort.InferenceSession(str(p2), providers=["CUDAExecutionProvider"])
outs = [o.name for o in s.get_outputs()]; ins = {i.name for i in s.get_inputs()}
r = dict(zip(outs, s.run(outs, {k: v for k, v in feeds.items() if k in ins})))
print("RESULT fixed-seed cuda vs cpu maxdiff", float(np.abs(r["hidden_in_grad"] - c["hidden_in_grad"]).max()))
