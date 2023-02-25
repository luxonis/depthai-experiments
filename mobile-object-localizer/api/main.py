#!/usr/bin/env python3
import argparse
import time

import blobconverter
import cv2
import depthai as dai
import numpy as np

'''
Mobile object localizer demo running on device on RGB camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Link to the original model:
https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

Blob taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Coonfidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH = blobconverter.from_zoo(name="mobile_object_localizer_192x192", zoo_type="depthai", shaves=6)
NN_WIDTH = 192
NN_HEIGHT = 192
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


# --------------- Methods ---------------

def plot_boxes(frame, boxes, colors, scores):
    color_black = (0, 0, 0)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        y1 = (frame.shape[0] * box[0]).astype(int)
        y2 = (frame.shape[0] * box[2]).astype(int)
        x1 = (frame.shape[1] * box[1]).astype(int)
        x2 = (frame.shape[1] * box[3]).astype(int)
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
        cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)


# --------------- Pipeline ---------------
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(NN_PATH)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)

# Create manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manip.initialConfig.setKeepAspectRatio(False)

# Link preview to manip and manip to nn
cam.preview.link(manip.inputImage)
manip.out.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)
cam.preview.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
detection_nn.out.link(xout_nn.input)

xout_manip = pipeline.create(dai.node.XLinkOut)
xout_manip.setStreamName("manip")
xout_manip.input.setBlocking(False)
manip.out.link(xout_manip.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    np.random.seed(0)
    colors_full = np.random.randint(255, size=(100, 3), dtype=int)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()
        in_manip = q_manip.get()

        frame = in_cam.getCvFrame()
        frame_manip = in_manip.getCvFrame()
        frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)

        # get outputs
        detection_boxes = np.array(in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
        detection_scores = np.array(in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))

        # keep boxes bigger than threshold
        mask = detection_scores >= THRESHOLD
        boxes = detection_boxes[mask]
        colors = colors_full[mask]
        scores = detection_scores[mask]

        # draw boxes
        plot_boxes(frame, boxes, colors, scores)
        plot_boxes(frame_manip, boxes, colors, scores)

        # show fps
        # show fps and predicted count
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # show frame
        cv2.imshow("Localizer", frame)
        cv2.imshow("Manip + NN", frame_manip)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
