import sys, tempfile, numpy as np, onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from construct.generate_artifact import generate_artifacts
from construct.loss import ReduceSum
import onnxruntime.training.onnxblock as onnxblock
from construct.onnx_block_utils import set_temp_file_names

B = int(sys.argv[1]); variant = sys.argv[2]
d = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
a_i = h.make_tensor_value_info("a", T.FLOAT, [B]); b_i = h.make_tensor_value_info("b", T.FLOAT, [B])
if variant == "div":
    nodes=[h.make_node("Div",["a","b"],["q"])]
else:
    nodes=[h.make_node("Reciprocal",["b"],["rb"]), h.make_node("Mul",["a","rb"],["q"])]
g = h.make_graph(nodes,"g",[a_i,b_i],[h.make_tensor_value_info("q", T.FLOAT, [B])])
m = h.make_model(g, opset_imports=[h.make_opsetid("",15)]); m.ir_version=8
onnx.save(m, d/"m.onnx")
class L(onnxblock.Block):
    def __init__(self):
        super().__init__(); self.r = ReduceSum(axes=[-1], keepdims=False); set_temp_file_names(self, d, "loss")
    def build(self, q): return self.r(q)
p = generate_artifacts(d/"m.onnx", save_directory=d, requires_grad=["a","b"], loss=L(), loss_input_names=["q"], gradient_model_name="g.onnx")
rng = np.random.default_rng(0)
a = rng.uniform(1,2,B).astype(np.float32); b = rng.uniform(1,2,B).astype(np.float32)
exp = {"a":1/b, "b":-a/b**2}
for prov in ["CPUExecutionProvider","CUDAExecutionProvider"]:
    s = ort.InferenceSession(str(p), providers=[prov])
    outs=[o.name for o in s.get_outputs()]
    r = dict(zip(outs, s.run(outs, {"a":a,"b":b})))
    for k,e in exp.items():
        nm=[o for o in outs if o.startswith(k) and "grad" in o or o==f"{k}_grad"]
        got=r[nm[0]]; print("RESULT",variant,B,prov,k,"maxerr",float(np.abs(got-e).max()),flush=True)
