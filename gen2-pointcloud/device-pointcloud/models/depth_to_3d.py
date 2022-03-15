#! /usr/bin/env python3

import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter
import os

def depth_to_3d(depth: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1
    points_3d: torch.Tensor = xyz * points_depth
    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW

class Model(nn.Module):
    def forward(self, xyz, depth):
        # TODO: once U16 -> FP16 is supported, use that.
        depthFP16 = 256.0 * depth[:,:,:,1::2] + depth[:,:,:,::2]
        return depth_to_3d(depthFP16, xyz)

def createBlob(resolution, path, name):
    (width, heigth) = resolution
    onnx_path = str(path / (name + '.onnx'))
    return_path = str(path / (name + '.blob'))

    # Define the expected input shape (dummy input)
    # Note there are twice as many columns as in the actual image because the network will interpret the memory buffer input as as uint8
    # even though it is actually uint16.
    shape = (1, 1, heigth, width * 2)
    model = Model()
    depth = torch.ones(shape, dtype=torch.float16)
    xyz = torch.ones((1, heigth, width, 3), dtype=torch.float16)

    torch.onnx.export(
        model,
        (xyz, depth),
        onnx_path,
        input_names = ['xyz', 'depth'], # Optional
        output_names = ['out'],
        opset_version=12,
        do_constant_folding=True,
    )

    onnx_simplified_path = str(path / (name + '_simplified.onnx'))

    # Use onnx-simplifier to simplify the onnx model
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, onnx_simplified_path)

    # Use blobconverter to convert onnx->IR->blob
    blob_path = blobconverter.from_onnx(
        model=onnx_simplified_path,
        data_type="FP16",
        shaves=6,
        use_cache=False,
        output_dir="out",
        optimizer_params=[],
        # XYZ is passed as FP16
        compile_params=['-iop xyz:FP16,depth:U8'],
    )

    os.rename(blob_path, return_path)
    return return_path