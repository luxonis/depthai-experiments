#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter
from kornia.geometry.depth import depth_to_3d
import os

class Model(nn.Module):
    def forward(self, matrix, depth):
        # TODO: once U16 -> FP16 is supported, use that.
        depthFP16 = 256.0 * depth[:,:,:,1::2] + depth[:,:,:,::2]
        print("depthFP16", depthFP16)

        # TODO: Use FP16 camera matrix
        matrixFP16 = 256.0 * matrix[1::2] + matrix[::2]
        print("a", matrixFP16)
        cam_matrix = torch.reshape(matrixFP16, (1,3,3))
        print("b", cam_matrix)
        matrix_ret = matrixFP16 * 1
        return depth_to_3d(depthFP16, cam_matrix, normalize_points=False), matrix_ret

def getPath(resolution):
    (width, heigth) = resolution
    path = Path("models", "out")
    path.mkdir(parents=True, exist_ok=True)
    name = f"pointcloud_{width}x{heigth}"

    onnx_path = str(path / (name + '.onnx'))
    return_path = str(path / (name + '.blob'))

    # if os.path.exists(return_path):
        # return return_path
    # Otherwise generate the model

    # Define the expected input shape (dummy input)
    # Note there are twice as many columns as in the actual image because the network will interpret the memory buffer input as as uint8
    # even though it is actually uint16.
    shape = (1, 1, heigth, width * 2)
    model = Model()
    X = torch.ones(shape, dtype=torch.float16)
    camMatrix = torch.ones((18), dtype=torch.float16)
    model.forward(camMatrix, X)

    torch.onnx.export(
        model,
        (camMatrix, X),
        onnx_path,
        input_names = ['matrix', 'depth'], # Optional
        output_names = ['out', 'cameramatrix'], # Optional
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
        compile_params=['-iop matrix:U8,depth:U8'],
    )

    os.rename(blob_path, return_path)
    return return_path