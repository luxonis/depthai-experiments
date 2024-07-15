#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import argparse
import errno
import blobconverter
import os
from yolop import YoloP
'''
YOLOP demo running on device with video input from host.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -v path/to/video

ONNX is taken from:
https://github.com/hustvl/YOLOP

Blob was compiled using OpenVino 2021.4
'''

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video_path', help="Path to video frame", default="vids/1.mp4")
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.5, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.3, type=float)

args = parser.parse_args()

VIDEO_SOURCE = args.video_path
CONFIDENCE_THRESHOLD = args.confidence_thresh
IOU_THRESHOLD = args.iou_thresh

NN_WIDTH = 320
NN_HEIGHT = 320

IR_WIDTH = 640
IR_HEIGHT = 360

NN_PATH = str(blobconverter.from_zoo(name="yolop_320x320", zoo_type="depthai", shaves=7))

vid_path = Path(VIDEO_SOURCE)
if not vid_path.is_file():
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), VIDEO_SOURCE)

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(vid_path)
    replay.setLoop(False)
    replay.setSize(NN_WIDTH, NN_HEIGHT)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(NN_PATH)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)
    replay.out.link(detection_nn.input)

    yolop = pipeline.create(YoloP).build(replay.out, detection_nn.out, NN_HEIGHT, NN_WIDTH)
    yolop.set_confidence_threshold(CONFIDENCE_THRESHOLD)
    yolop.set_iou_threshold(IOU_THRESHOLD)
    
    pipeline.run()