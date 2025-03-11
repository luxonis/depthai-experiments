import os

import depthai as dai
import kornia
from generate_utils import generate_model, parse_arguments
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from torch import nn

args = parse_arguments()

platform = dai.Platform.__members__[args.platform.upper()]


class Model(nn.Module):
    def forward(self, image):
        return kornia.filters.laplacian(
            image, kernel_size=3, border_type="reflect", normalized=True
        )


model = Model()

input_shape = (1, 3, 300, 300)

cfg_dict = {
    "config_version": CONFIG_VERSION,
    "model": {
        "metadata": {
            "name": "edge_simplified",
            "path": "edge_simplified.onnx",
            "precision": "float32",
        },
        "inputs": [
            {
                "name": "input_img",
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
            }
        ],
        "outputs": [{"name": "output_img", "dtype": "float32"}],
        "heads": [
            {
                "parser": "ImageOutputParser",
                "outputs": ["output_img"],
                "metadata": {"output_is_bgr": True},
            }
        ],
    },
}

os.makedirs(str("out/models"), exist_ok=True)
generate_model(
    model=model,
    cfg_dict=cfg_dict,
    output_path=f"out/models/edge.{args.platform.lower()}.tar.xz",
    simplify_model=True,
    platform=platform,
)
