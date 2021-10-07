#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import blobconverter

class Model(nn.Module):
    def forward(self, img: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor):
        # output = (input - mean) / scale
        return torch.div(torch.sub(img, mean), scale)

# Define the expected input shape (dummy input)
shape = (3, 300, 300)
X = torch.ones(shape, dtype=torch.float32)
Arg = torch.ones((1,1,1), dtype=torch.float32)

path = Path("out/")
path.mkdir(parents=True, exist_ok=True)
onnx_file = "out/normalize.onnx"

print(f"Writing to {onnx_file}")
torch.onnx.export(
    Model(),
    (X, Arg, Arg),
    onnx_file,
    opset_version=12,
    do_constant_folding=True,
    input_names = ['frame', 'scale', 'mean'], # Optional
    output_names = ['output'], # Optional
)

# No need for onnx-simplifier here

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_file,
    data_type="FP16",
    shaves=4,
    use_cache=False,
    output_dir="../models",
    optimizer_params=[],
    compile_params = ["--ip=FP16"]
)