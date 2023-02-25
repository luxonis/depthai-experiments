#!/usr/bin/env python3

import argparse
import time

import cv2
import depthai as dai
import numpy as np

from utils.effect import EffectRenderer2D

'''
MediaPipe Facial Landmark detector with PNG EffectRenderer.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -conf [CONF]

Blob is converted from MediaPipe's tflite model.
'''

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.5, type=float)

args = parser.parse_args()
CONF_THRESH = args.confidence_thresh
NN_PATH = "models/face_landmark_openvino_2021.4_6shave.blob"
NN_WIDTH, NN_HEIGHT = 192, 192
PREVIEW_WIDTH, PREVIEW_HEIGHT = 416, 416
OVERLAY_IMAGE = "mask/facepaint.png"

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
cam.setFps(30)

# Neural Network for Landmark detection
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(NN_PATH)
nn.setNumPoolFrames(4)
nn.input.setBlocking(False)
nn.setNumInferenceThreads(2)

# Image Manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResizeThumbnail(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)

# Create outputs

xin_rgb = pipeline.create(dai.node.XLinkIn)
xin_rgb.setStreamName("xin")

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

# Linking
cam.preview.link(xout_rgb.input)
cam.preview.link(manip.inputImage)
manip.out.link(nn.input)
nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues
    q_cam = device.getOutputQueue(name="cam", maxSize=4)
    q_nn = device.getOutputQueue(name="nn", maxSize=4)

    # Read face overlay
    effect_rendered = EffectRenderer2D(OVERLAY_IMAGE)

    # FPS Init
    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()

        # Get frame
        frame = in_cam.getCvFrame()

        # Get outputs
        score = np.array(in_nn.getLayerFp16('conv2d_31')).reshape((1,))
        score = 1 / (1 + np.exp(-score[0]))  # sigmoid on score
        landmarks = np.array(in_nn.getLayerFp16('conv2d_21')).reshape((468, 3))

        if score > CONF_THRESH:
            # scale landmarks
            ldms = landmarks  # .copy()
            ldms *= np.array([PREVIEW_WIDTH / NN_WIDTH, PREVIEW_HEIGHT / NN_HEIGHT, 1])

            # render frame
            target_frame = frame.copy()
            applied_effect = effect_rendered.render_effect(target_frame, ldms)
            cv2.imshow("Effect", applied_effect)

            # show landmarks on frame
            for ldm in ldms:
                col = (0, 0, int(ldm[2]) * 5 + 100)
                cv2.circle(frame, (int(ldm[0]), int(ldm[1])), 1, col, 1)

        else:
            applied_effect = frame

        cv2.imshow("Demo", np.hstack([frame, applied_effect]))

        # Show FPS and score
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        label_count = "Score: {:.2f}".format(score)

        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        (w2, h2), _ = cv2.getTextSize(label_count, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)

        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.rectangle(frame, (0, 0), (w2 + 2, h2 + 6), color_white, -1)

        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        cv2.putText(frame, label_count, (2, 12), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        cv2.imshow("Landmarks", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
