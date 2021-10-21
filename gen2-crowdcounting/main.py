#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import errno
import os

'''
DM-Count crowd counting demo running on device with video input from host.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -v path/to/video

Model is taken from:
https://github.com/cvlab-stonybrook/DM-Count
and weights from:
https://github.com/tersekmatija/lwcc

DepthAI 2.9.0.0 is required. Blob was compiled using OpenVino 2021.4
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/vgg_openvino_2021.4_6shave.blob', type=str)

args = parser.parse_args()
nn_path = args.nn_model 

# resize input to smaller size for faster inference
NN_WIDTH = 426
NN_HEIGHT = 240


# --------------- Methods ---------------
# scale to [0 ... 255] and apply colormap
def decode_density_map(output_tensor):
    output = np.array(output_tensor) * 255
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_VIRIDIS)
    output_colors = cv2.resize(output_colors, (NN_WIDTH, NN_HEIGHT), interpolation = cv2.INTER_LINEAR)
    return output_colors


# merge 2 frames together
def show_output(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.5, 0)


# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Create Manip for image resizing and NN for count inference
detection_nn = pipeline.createNeuralNetwork()

camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(NN_WIDTH, NN_HEIGHT)

# Create output links, and in link for video
manipOut = pipeline.createXLinkOut()
xlinkOut = pipeline.createXLinkOut()
nnOut = pipeline.createXLinkOut()

manipOut.setStreamName("manip")
xlinkOut.setStreamName("trackerFrame")
nnOut.setStreamName("nn")

# setting node configs
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Linking
camRgb.preview.link(manipOut.input)
camRgb.preview.link(detection_nn.input)
camRgb.setInterleaved(False)
detection_nn.out.link(nnOut.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    qManip = device.getOutputQueue(name="manip", maxSize=4)
    qNN = device.getOutputQueue(name="nn", maxSize=4)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None


    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    while True:
        # get resized image and NN output queues
        manip = qManip.get()
        inNN = qNN.get()

        # read output
        lay1 = np.array(inNN.getFirstLayerFp16()).reshape((30,52))  # density map output is 1/8 of input size
        count = np.sum(lay1)    # predicted count is the sum of the density map
        manipFrame = manip.getCvFrame()

        # show fps and predicted count
        color_black, color_white = (0,0,0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        label_count = "Predicted count: {:.2f}".format(count)

        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        (w2, h2), _ = cv2.getTextSize(label_count, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)

        cv2.rectangle(manipFrame, (0,manipFrame.shape[0]-h1-6), (w1 + 2, manipFrame.shape[0]), color_white, -1)
        cv2.rectangle(manipFrame, (0,0), (w2 + 2, h2 + 6), color_white, -1)

        cv2.putText(manipFrame, label_fps, (2, manipFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        cv2.putText(manipFrame, label_count, (2, 12), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # generate and append density map
        output_colors = decode_density_map(lay1)
        outFrame = show_output(output_colors, manipFrame)
        cv2.imshow("Predict count", outFrame)

        # FPS counter
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        if cv2.waitKey(1) == ord('q'):
            break
