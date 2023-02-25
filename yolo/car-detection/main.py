import argparse
import json
from pathlib import Path

from depthai_sdk import OakCamera

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model path for inference",
                    default='models/yolo_v4_tiny_openvino_2021.3_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='yolo-tiny.json', type=str)
args = parser.parse_args()

with OakCamera(replay='cars-tracking-above-01') as oak:
    color = oak.create_camera('color')

    nn = oak.create_nn(args.model, color, nn_type='yolo')

    with open(str(Path(args.config).resolve())) as file:
        conf = json.load(file)
        nn.config_yolo_from_metadata(conf['nn_config']['NN_specific_metadata'])

    oak.visualize(nn.out.passthrough, fps=True)  # 1080P -> 720P
    oak.start(blocking=True)
