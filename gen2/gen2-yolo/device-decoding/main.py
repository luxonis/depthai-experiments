from depthai_sdk import OakCamera, ArgsManager
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Provide json config path for inference", default='json/yolov4-tiny.json', type=str)
args = ArgsManager.parseArgs(parser)

with OakCamera() as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn(args.config, color, out='yolo', type='yolo')
    oak.create_visualizer([color, nn], fps=True)
    oak.start(blocking=True)
