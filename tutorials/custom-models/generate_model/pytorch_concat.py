import os

import depthai as dai
import torch
from generate_utils import generate_model, parse_arguments
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from torch import nn

args = parse_arguments()

platform = dai.Platform.__members__[args.platform.upper()]


class CatImgs(nn.Module):
    def forward(self, img1, img2, img3):
        return torch.cat((img1, img2, img3), 3)


model = CatImgs()

input_shape = (1, 3, 300, 300)

cfg_dict = {
    "config_version": CONFIG_VERSION,
    "model": {
        "metadata": {
            "name": "concat",
            "path": "concat.onnx",
            "precision": "float32",
        },
        "inputs": [
            {
                "name": "img1",
                "dtype": "float32",
                "input_type": "image",
                "shape": input_shape,
                "layout": "NCHW",
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
                "layout": "NCHW",
                "preprocessing": {
                    "mean": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "reverse_channels": False,
                    "interleaved_to_planar": None,
                },
            },
            {
                "name": "img3",
                "dtype": "float32",
                "input_type": "image",
                "shape": input_shape,
                "layout": "NCHW",
                "preprocessing": {
                    "mean": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "reverse_channels": False,
                    "interleaved_to_planar": None,
                },
            },
        ],
        "outputs": [{"name": "output", "dtype": "float32"}],
        "heads": [
            {
                "parser": "ImageOutputParser",
                "outputs": ["output"],
                "metadata": {"output_is_bgr": True},
            }
        ],
    },
}

os.makedirs(str("out/models"), exist_ok=True)
generate_model(
    model=model,
    cfg_dict=cfg_dict,
    output_path=f"out/models/concat.{args.platform.lower()}.tar.xz",
    simplify_model=False,
    platform=platform,
)
