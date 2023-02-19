import torch
import torch.nn as nn
import numpy as np
import os
import subprocess
import blobconverter
import argparse

from anomalib.deploy import TorchInferencer
from anomalib.deploy import export, ExportMode

parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint', type=str, help="Path to PADIM checkpoint", required=True)
parser.add_argument('-n', '--name', type=str, help="Name of the exported model", default="padim")
args = parser.parse_args()

class MyPadim(nn.Module):

    def __init__(self, model, meta_data):
        super(MyPadim, self).__init__()
        self.model = model
        self.meta_data = meta_data

    def normalize_min_max(self, targets, threshold, min_val, max_val):
        normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
        return normalized

    def forward(self, X):
        anomaly_map = self.model.forward(X)
        pred_score = torch.max(anomaly_map)

        anomaly_map = self.normalize_min_max(
            anomaly_map, self.meta_data["pixel_threshold"], self.meta_data["min"], self.meta_data["max"]
        )
        pred_score = self.normalize_min_max(
            pred_score, self.meta_data["image_threshold"], self.meta_data["min"], self.meta_data["max"]
        )

        return anomaly_map, pred_score

inferencer = TorchInferencer('anomalib/models/padim/config.yaml',
                             args.checkpoint)

model = MyPadim(inferencer.model, inferencer.meta_data)
dummy = torch.randn(1,3,256,256)

if not os.path.exists('export/'):
    os.mkdir('export/')

print("Converting PyTorch model to ONNX")
torch.onnx.export(
    model,
    dummy,
    f"export/{args.name}.onnx",
    opset_version=11,
    verbose=True,
    input_names=["input"],
    output_names=["anomaly_map","pred_score"]
)

print("Converting ONNX to IR")
cmd = f"mo \
--input_model export/{args.name}.onnx \
--model_name {args.name} \
--mean_values [123.675,116.28,103.53] \
--scale_values [58.395,57.12,57.375] \
--data_type FP16 \
--output_dir export/"

subprocess.check_output(cmd, shell=True)

blob_path = blobconverter.from_openvino(
    xml=f"export/{args.name}.xml",
    bin=f"export/{args.name}.bin",
    data_type="FP16",
    shaves=6,
    output_dir='./export/'
)

os.rename(blob_path, f"./export/{args.name}.blob")
