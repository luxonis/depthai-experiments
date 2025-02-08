#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/model.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape = 256

def decode_deeplabv3p(output):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    # output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame, weight=0.2):
    return cv2.addWeighted(frame, 1, output_colors,weight,0)



# Start defining a pipeline
pipeline = dai.Pipeline()

# pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(nn_shape,nn_shape)
cam.setFp16(True)
cam.setInterleaved(False)
cam.preview.link(detection_nn.input)
cam.setFps(50)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False

while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
        data = in_nn_input.getData()
        # TODO: FIx this mess
        frame = np.array(data).astype(np.uint8).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow("rgb", frame)
        # frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        # print("NN received")
        layer1 = in_nn.getFirstLayerFp16()
            # reshape to numpy array
        lay1 = np.asarray(layer1, dtype=np.float16).reshape((nn_shape, nn_shape))

        # print(lay1)

        newMatrix = np.array(lay1, dtype=np.int32)
        # print(newMatrix)
        output_colors = decode_deeplabv3p(newMatrix)
        if frame is not None:
            cv2.imshow("selfie", show_deeplabv3p(output_colors, frame, 1.0))
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (20,20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            cv2.imshow("nn_input", frame)

    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()


    if cv2.waitKey(1) == ord('q'):
        break
