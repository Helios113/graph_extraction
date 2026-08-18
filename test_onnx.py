import torch
import onnx
import onnxruntime as ort
import numpy as np
import tempfile
import os
from construct.pullback import generate_subgraph_pullback
from config import LossType
from pathlib import Path
from execute.jacobian import compute_jacobian

class Pullback_Test(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear1= torch.nn.Linear(2,2,False)
        self.gelu = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(2,2,False)
        self.softmax = torch.nn.Softmax(dim=-1)     
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
        
        

def forward_test(model, x):
    with torch.no_grad():
        torch_out = model(x)
    
    # with tempfile.TemporaryDirectory() as tmpdir:
        # onnx_path = os.path.join(tmpdir, "test_model.onnx")
        onnx_path  = "test_model.onnx"
        torch.onnx.export(model, x, f=onnx_path)

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: x.numpy()})[0]

        torch_out_np = torch_out.numpy() if isinstance(torch_out, torch.Tensor) else torch_out.detach().numpy()
        max_diff = np.abs(torch_out_np - onnx_out).max()

        print(f"Max abs diff: {max_diff}")
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-5)
        print("Outputs match within tolerance.")
            
            
def pullback_test(model, x):
    
    with torch.no_grad():
        torch_out = model(x)
    
    # with tempfile.TemporaryDirectory() as tmpdir:
        # onnx_path = os.path.join(tmpdir, "test_model.onnx")
        onnx_path  = "test_model.onnx"
        torch.onnx.export(model, x, f=onnx_path)

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: x.numpy()})[0]

        torch_out_np = torch_out.numpy() if isinstance(torch_out, torch.Tensor) else torch_out.detach().numpy()
        max_diff = np.abs(torch_out_np - onnx_out).max()

        print(f"Max abs diff: {max_diff}")
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-5)
        print("Outputs match within tolerance.")
        paths = generate_subgraph_pullback(
            Path(onnx_path),
            LossType.MaskedSumLoss,
            [10, 2],
            "x",
            "softmax",
        )
        model.eval()
        x = x.clone().requires_grad_(True)
        torch_out = model(x)

        # Full Jacobian d(output)/d(input): shape [10, 2, 10, 2] since output is [10, 2]
        jacobian = torch.autograd.functional.jacobian(model, x)
        
        print(jacobian.shape)
        
        onnx_model_jac = paths["gradient"]
        jac_sess = ort.InferenceSession(str(onnx_model_jac))

        
        jac_out = compute_jacobian(
            jac_sess,
            input_data = x.detach().numpy(),
            input_name = "x",
            output_name= "softmax",
            mask_name = "mask",
            gradient_output_name = "x_grad",
            output_shape=[10,2],
        )
        print(jac_out.shape)

        max_diff = np.abs(jacobian - jac_out).max()

        print(f"Max abs diff: {max_diff}")
        np.testing.assert_allclose(jacobian, jac_out, rtol=1e-3, atol=1e-5)
        print("Outputs match within tolerance.")

if __name__ == "__main__":
    torch.manual_seed(0)
    model = Pullback_Test()
    model.eval()
    x = torch.rand([10, 2])
    
    pullback_test(model, x)
    
    # pullback_test

   