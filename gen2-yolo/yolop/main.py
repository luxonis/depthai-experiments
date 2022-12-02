#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import errno
import blobconverter
import os
from utils.functions import non_max_suppression, show_masks, show_boxes

'''
YOLOP demo running on device with video input from host.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -v path/to/video

ONNX is taken from:
https://github.com/hustvl/YOLOP

DepthAI 2.11.0.0 is required. Blob was compiled using OpenVino 2021.4
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video_path', help="Path to video frame", default="vids/1.mp4")
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.5, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.3, type=float)

args = parser.parse_args()

VIDEO_SOURCE = args.video_path
#NN_PATH = args.nn_model
CONFIDENCE_THRESHOLD = args.confidence_thresh
IOU_THRESHOLD = args.iou_thresh

# resize input to smaller size for faster inference
NN_WIDTH = 320
NN_HEIGHT = 320

# set initial resize so the input is not too large
IR_WIDTH = 640
IR_HEIGHT = 360

# --------------- Get Blob ------------------
NN_PATH = str(blobconverter.from_zoo(name="yolop_320x320", zoo_type="depthai", shaves = 7))

# --------------- Check input ---------------
vid_path = Path(VIDEO_SOURCE)
if not vid_path.is_file():
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), VIDEO_SOURCE)

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Create Manip for image resizing and NN for count inference
manip = pipeline.create(dai.node.ImageManip)
detection_nn = pipeline.create(dai.node.NeuralNetwork)

# Create output links, and in link for video
xout_manip = pipeline.create(dai.node.XLinkOut)
xin_frame = pipeline.create(dai.node.XLinkIn)
xout_nn = pipeline.create(dai.node.XLinkOut)

xout_manip.setStreamName("manip")
xin_frame.setStreamName("in_frame")
xout_nn.setStreamName("nn")

# Properties
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.inputImage.setBlocking(True)

# setting node configs
detection_nn.setBlobPath(NN_PATH)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Linking
manip.out.link(xout_manip.input)
manip.out.link(detection_nn.input)
xin_frame.out.link(manip.inputImage)
detection_nn.out.link(xout_nn.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    q_in = device.getInputQueue(name="in_frame", maxSize=1, blocking=False)
    q_manip = device.getOutputQueue(name="manip", maxSize=4)
    q_nn = device.getOutputQueue(name="nn", maxSize=4)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None


    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    cap = cv2.VideoCapture(args.video_path)
    base_ts = time.monotonic()
    simulated_fps = 30
    input_frame_shape = (IR_WIDTH, IR_HEIGHT)

    while cap.isOpened():
        # read, process image and send it to NN
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.RGB888p)
        img.setData(to_planar(frame, input_frame_shape))
        img.setTimestamp(base_ts)
        base_ts += 1 / simulated_fps

        img.setWidth(input_frame_shape[0])
        img.setHeight(input_frame_shape[1])
        q_in.send(img)

        # get resized image and NN output queues
        manip = q_manip.get()
        manip_frame = manip.getCvFrame()

        in_nn = q_nn.get()

        # read output
        area = np.array(in_nn.getLayerFp16("drive_area_seg")).reshape((1, 2, NN_HEIGHT, NN_WIDTH))
        lines = np.array(in_nn.getLayerFp16("lane_line_seg")).reshape((1, 2, NN_HEIGHT, NN_WIDTH))
        dets = np.array(in_nn.getLayerFp16("det_out")).reshape((1, 6300, 6))

        # generate and append density map
        boxes = np.array(non_max_suppression(dets, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)[0])
        show_boxes(manip_frame, boxes)
        show_masks(manip_frame, area, lines)

        # show fps and predicted count
        color_black, color_white = (0,0,0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(manip_frame, (0,manip_frame.shape[0]-h1-6), (w1 + 2, manip_frame.shape[0]), color_white, -1)
        cv2.putText(manip_frame, label_fps, (2, manip_frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color_black)

        cv2.imshow("Predict count", manip_frame)

        # FPS counter
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        if cv2.waitKey(1) == ord('q'):
            break