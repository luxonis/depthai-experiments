#!/usr/bin/env python3

import argparse
import time

import blobconverter
import cv2
import depthai as dai
import numpy as np

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb',
                    choices=cam_options)
parser.add_argument("-s", "--size", help="Shape", choices=['256', '513'], default='256', type=str)
args = parser.parse_args()

cam_source = args.cam_input
nn_shape = int(args.size)


def decode_deeplabv3p(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape, nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)


# Start defining a pipeline
pipeline = dai.Pipeline()
detection_nn = pipeline.create(dai.node.NeuralNetwork)
nn_path = blobconverter.from_zoo(name=f"deeplab_v3_mnv2_{args.size}x{args.size}", zoo_type="depthai", shaves=6)
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
with dai.Device(pipeline.getOpenVINOVersion()) as device:
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

        if in_nn_input is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
            frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if in_nn is not None:
            # print("NN received")
            layers = in_nn.getAllLayers()

            if not layer_info_printed:
                for layer_nr, layer in enumerate(layers):
                    print(f"Layer {layer_nr}")
                    print(f"Name: {layer.name}")
                    print(f"Order: {layer.order}")
                    print(f"dataType: {layer.dataType}")
                    dims = layer.dims[::-1]  # reverse dimensions
                    print(f"dims: {dims}")
                layer_info_printed = True

            # get layer1 data
            layer1 = in_nn.getLayerInt32(layers[0].name)
            # reshape to numpy array
            dims = layer.dims[::-1]
            lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)

            output_colors = decode_deeplabv3p(lay1)

            if frame is not None:
                frame = show_deeplabv3p(output_colors, frame)
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            (255, 0, 0))
                cv2.imshow("nn_input", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
