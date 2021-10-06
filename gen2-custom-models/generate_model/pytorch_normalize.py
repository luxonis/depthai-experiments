#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import blobconverter

class Model(nn.Module):
    def forward(self, img: torch.Tensor, mul: torch.Tensor, add: torch.Tensor):
        mul = torch.mul(img, mul)
        return torch.add(mul, add)

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
    input_names = ['frame', 'multiplier', 'addend'], # Optional
    output_names = ['output'], # Optional
)

# No need for onnx-simplifier here

# Use blobconverter to convert onnx->IR->blob
blobconverter.from_onnx(
    model=onnx_file,
    data_type="FP16",
    shaves=6,
    use_cache=False,
    output_dir="../models",
    optimizer_params=[]
)