#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time

usb2mode=True
classes_nr = 13

# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setPipelineOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2020_1)

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(513, 513)
cam_rgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/saved_model.blob')).resolve().absolute()))
detection_nn.setNumPoolFrames(4)
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
device = dai.Device(pipeline, usb2Mode=usb2mode)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
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
        print("NN received")
        output_layers = in_nn.getAllLayerNames()
        print(output_layers)
        # layers=in_nn.getAllLayers()
        layer1 = in_nn.getLayerInt32(output_layers[0])

        lay1 = np.asarray(layer1, dtype=np.int32).reshape((1,513,513))
        print(lay1.shape)

        layer2 = in_nn.getLayerFp16(output_layers[1])
        lay2 = np.asarray(layer2, dtype=np.int32).reshape((1,classes_nr,513,513))
        print(lay2.shape)


    if frame is not None:
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("rgb", frame)

    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break
