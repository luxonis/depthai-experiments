#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np


def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    output = output_tensor.reshape(256,256)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.2,0)



# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setPipelineOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2020_1)

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(256, 256)
cam_rgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/deeplabv3p_person.blob.sh13cmx13NCE1')).resolve().absolute()))
detection_nn.setNumPoolFrames(1)
detection_nn.input.setBlocking(False)
# detection_nn.setNumInferenceThreads(1)

cam_rgb.preview.link(detection_nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
xout_rgb.input.setBlocking(False)

# cam_rgb.preview.link(xout_rgb.input)
detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)


while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_rgb = q_rgb.get()
    in_nn = q_nn.get()

    if in_rgb is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        # print("NN received")
        output_layers = in_nn.getAllLayerNames()
        # print(output_layers)

        layer1 = in_nn.getLayerInt32(output_layers[0])

        lay1 = np.asarray(layer1, dtype=np.int32).reshape((1,256,256))
        # print(lay1.shape)

        output_colors = decode_deeplabv3p(lay1)

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
