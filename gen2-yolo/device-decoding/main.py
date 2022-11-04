from depthai_sdk import OakCamera, ArgsParser
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/yolo.json', type=str)
args = ArgsParser.parseArgs(parser)

with OakCamera(args=args) as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(args['config'], color, nnType='yolo', spatial=True)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)
