import os

import depthai as dai
import torch
from generate_utils import generate_model, parse_arguments
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from torch import nn

args = parse_arguments()

platform = dai.Platform.__members__[args.platform.upper()]


class Model(nn.Module):
    def forward(self, img1, img2):
        sum1 = torch.sum(img1, dim=0)
        sum2 = torch.sum(img2, dim=0)
        return torch.sub(sum1, sum2)


model = Model()

input_shape = (3, 720, 720)

cfg_dict = {
    "config_version": CONFIG_VERSION,
    "model": {
        "metadata": {
            "name": "diff",
            "path": "diff.onnx",
            "precision": "float32",
        },
        "inputs": [
            {
                "name": "img1",
                "dtype": "float32",
                "input_type": "image",
                "shape": input_shape,
                "layout": "CHW",
                "preprocessing": {
                    "mean": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "reverse_channels": False,
                    "interleaved_to_planar": None,
                },
            },
            {
                "name": "img2",
                "dtype": "float32",
                "input_type": "image",
                "shape": input_shape,
                "layout": "CHW",
                "preprocessing": {
                    "mean": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "reverse_channels": False,
                    "interleaved_to_planar": None,
                },
            },
        ],
        "outputs": [{"name": "output", "dtype": "float32"}],
        "heads": [],
    },
}

os.makedirs(str("out/models"), exist_ok=True)
generate_model(
    model=model,
    cfg_dict=cfg_dict,
    output_path=f"out/models/diff.{args.platform.lower()}.tar.xz",
    simplify_model=False,
    platform=platform,
)
