#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import blobconverter

class CatImgs(nn.Module):
    def forward(self, img1, img2, img3):
        return torch.cat((img1, img2, img3), 3)

# Define the expected input shape (dummy input)
shape = (1, 3, 300, 300)
X = torch.ones(shape, dtype=torch.float32)

path = Path("out/")
path.mkdir(parents=True, exist_ok=True)
onnx_file = "out/concat.onnx"

print(f"Writing to {onnx_file}")
torch.onnx.export(
    CatImgs(),
    (X, X, X),
    onnx_file,
    opset_version=12,
    do_constant_folding=True,
    input_names = ['img1', 'img2', 'img3'], # Optional
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