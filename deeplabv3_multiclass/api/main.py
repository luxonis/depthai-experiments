#!/usr/bin/env python3

import argparse
import time

import cv2
import depthai as dai
import numpy as np

'''
Deeplabv3 multiclass running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob is taken from ML training examples:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks

You can clone the DeepLabV3plus_MNV2.ipynb notebook and try training the model yourself.

'''

num_of_classes = 21  # define the number of classes in the dataset
cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb',
                    choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference",
                    default='models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape = 256


def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(nn_shape, nn_shape)

    # scale to [0 ... 2555] and apply colormap
    output = np.array(output) * (255 / num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[output == 0] = [0, 0, 0]

    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.4, 0)


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam = None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(nn_shape, nn_shape)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.create(dai.node.ImageManip)
    manip.setResize(nn_shape, nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(40)

# Create outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if cam_source != "rgb" and not depth_enabled:
        raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
    device.startPipeline(pipeline)

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

        frame = in_nn_input.getCvFrame()

        layers = in_nn.getAllLayers()

        # get layer1 data
        lay1 = np.array(in_nn.getFirstLayerInt32()).reshape(nn_shape, nn_shape)

        found_classes = np.unique(lay1)
        output_colors = decode_deeplabv3p(lay1)

        frame = show_deeplabv3p(output_colors, frame)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 0, 0))
        cv2.putText(frame, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 0, 0))
        cv2.imshow("nn_input", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
