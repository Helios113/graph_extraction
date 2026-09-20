from config import load_config
from construct.construct import ensure_base_model, ensure_extra_outputs

cfg = load_config("configs/simple_stories.toml")

print(f"Config dir: {cfg.dir}")
print(f"Base model path: {cfg.base_model_path}")

ensure_base_model(cfg, verbose=True)

extra_outputs = ["embedding", "add_6", "add_8"]
out_path = ensure_extra_outputs(cfg, extra_outputs, force=True)
print(f"Extra outputs model -> {out_path}")

import onnx
model = onnx.load(str(out_path), load_external_data=False)
print("Graph inputs:", [i.name for i in model.graph.input])
print("Graph outputs:", [o.name for o in model.graph.output])
print("Num nodes:", len(model.graph.node))
