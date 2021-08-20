#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time

nn_shape = 896, 512

def decode(packet):
    data = np.squeeze(to_tensor_result(packet)["L0317_ReWeight_SoftMax"])
    class_colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    indices = np.argmax(data, axis=0)
    output_colors = np.take(class_colors, indices, axis=0)
    return output_colors


def draw(data, frame):
    if len(data) == 0:
        return
    cv2.addWeighted(frame, 1, cv2.resize(data, frame.shape[:2][::-1]), 0.2, 0, frame)


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


def to_tensor_result(packet):
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims)
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(*nn_shape)
cam.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(blobconverter.from_zoo(name='road-segmentation-adas-0001', shaves=6)))
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("cam")
detection_nn.passthrough.link(cam_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    frame = None
    depth_frame = None
    road_decoded = None

    while True:
        in_color = q_color.tryGet()

        if in_color is not None:
            fps.next_iter()
            frame = in_color.getCvFrame()
            road_decoded = decode(q_nn.get())

        if frame is not None:
            show_frame = frame.copy()
            if road_decoded is not None:
                cv2.addWeighted(show_frame, 1, cv2.resize(road_decoded, show_frame.shape[:2][::-1]), 0.2, 0, show_frame)

            cv2.putText(show_frame, "Fps: {:.2f}".format(fps.fps()), (2, show_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                        0.4, color=(255, 255, 255))
            cv2.imshow("weighted", show_frame)

        if cv2.waitKey(1) == ord('q'):
            break
