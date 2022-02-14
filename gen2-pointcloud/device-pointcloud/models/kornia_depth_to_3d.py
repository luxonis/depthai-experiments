#! /usr/bin/env python3

from pathlib import Path
import torch
from torch import nn
import onnx
from onnxsim import simplify
import blobconverter
from kornia.geometry.depth import depth_to_3d
import os
import math
import depthai as dai

class Model(nn.Module):
    # baseline in mm
    def setCameraCalib(self, calib, resolution):
        M_right = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
            dai.Size2f(resolution[0], resolution[1]),
        )
        self.camera_matrix = torch.Tensor([M_right])

        # Will convert disparity -> depth in NN, since INT16 isn't supported and can't yet be converted
        # self.baseline = calib.getBaselineDistance() / 100 # to get it in meters, 0.075m
        # fov = calib.getFov(dai.CameraBoardSocket.RIGHT)
        # self.focal_length_in_pixels = resolution[0] * 0.5 / math.tan(fov * 0.5 * math.pi/180)

    def forward(self, depth):
        # TODO: once ImageManip RAW16 -> GRAYF16 will be supported (on image_manip_refactor branch), use depht instead
        depthFP16 = 256.0 * depth[:,:,:,1::2] + depth[:,:,:,::2]
        return depth_to_3d(depthFP16, self.camera_matrix, normalize_points=False)

def getPath(mxid, resolution, calib):
    (width, heigth) = resolution
    path = Path("models", "out")
    path.mkdir(parents=True, exist_ok=True)
    name = f"{mxid}_pointcloud_{width}x{heigth}"

    onnx_path = str(path / (name + '.onnx'))
    return_path = str(path / (name + '.blob'))

    if os.path.exists(return_path):
        return return_path
    # Otherwise generate the model

    # Define the expected input shape (dummy input)
    # Note there are twice as many columns as in the actual image because the network will interpret the memory buffer input as as uint8
    # even though it is actually uint16.
    shape = (1, 1, heigth, width * 2)
    model = Model()
    model.setCameraCalib(calib, resolution)
    X = torch.ones(shape, dtype=torch.float16)

    # model.forward(X)
    # return ""

    torch.onnx.export(
        model,
        X,
        onnx_path,
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
        compile_params=["-ip U8"],
    )

    os.rename(blob_path, return_path)
    return return_path