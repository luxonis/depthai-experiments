#!/usr/bin/env python3

import time

import cv2
import depthai as dai
import numpy as np

'''
FastDepth demo running on device.
https://github.com/zl548/MegaDepth


Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Onnx taken from PINTO0309, added scaling flag, and exported to blob:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth
'''

# --------------- Arguments ---------------
nn_path = "models/megadepth_192x256_openvino_2021.4_6shave.blob"

# choose width and height based on model
NN_WIDTH, NN_HEIGHT = 256, 192

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Define a neural network
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# Link
cam.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_cam.input)
detection_nn.out.link(xout_nn.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()

        # Get output layer
        pred = np.array(in_nn.getFirstLayerFp16()).reshape((NN_HEIGHT, NN_WIDTH))

        # Scale depth to get relative depth
        d_min = np.min(pred)
        d_max = np.max(pred)
        depth_relative = (pred - d_min) / (d_max - d_min)

        # Color it
        depth_relative = np.array(depth_relative) * 255
        depth_relative = depth_relative.astype(np.uint8)
        depth_relative = 255 - depth_relative
        depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

        # Show FPS
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # Concatenate NN input and produced depth
        cv2.imshow("Detections", cv2.hconcat([frame, depth_relative]))

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
