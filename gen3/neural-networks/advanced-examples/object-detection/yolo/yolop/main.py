#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import argparse
import errno
import os
from yolop import YoloP
'''
YOLOP demo running on device with video input from host.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -v path/to/video
'''

model_description = dai.NNModelDescription(modelSlug="yolop", platform="RVC2", modelVersionSlug="320x320") # only for RVC2
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

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

vid_path = Path(VIDEO_SOURCE)
if not vid_path.is_file():
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), VIDEO_SOURCE)

with dai.Pipeline() as pipeline:
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(vid_path)
    replay.setLoop(False)
    replay.setSize(NN_WIDTH, NN_HEIGHT)
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setNNArchive(nn_archive)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)
    replay.out.link(detection_nn.input)

    yolop = pipeline.create(YoloP).build(
        img_frames=replay.out, 
        nn_data=detection_nn.out, 
        nn_height=NN_HEIGHT, 
        nn_width=NN_WIDTH
    )
    yolop.set_confidence_threshold(CONFIDENCE_THRESHOLD)
    yolop.set_iou_threshold(IOU_THRESHOLD)
    
    pipeline.run()