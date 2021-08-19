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


def dispay_colored_depth(frame, name):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
    cv2.imshow(name, frame_colored)
    return frame_colored


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
detection_nn.setBlobPath(str(blobconverter.from_zoo(name='road-segmentation-adas-0001')))
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

# Left mono camera
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(245)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create depth output
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    frame = None
    depth_frame = None
    road_decoded = None

    while True:
        in_color = q_color.tryGet()
        in_depth = q_depth.tryGet()

        if in_color is not None:
            fps.next_iter()
            frame = in_color.getCvFrame()
            road_decoded = decode(q_nn.get())

        if in_depth is not None:
            depth_frame = in_depth.getFrame()

        if frame is not None:
            show_frame = frame.copy()
            if road_decoded is not None:
                cv2.addWeighted(show_frame, 1, cv2.resize(road_decoded, show_frame.shape[:2][::-1]), 0.2, 0, show_frame)

            cv2.putText(show_frame, "Fps: {:.2f}".format(fps.fps()), (2, show_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                        0.4, color=(255, 255, 255))
            cv2.imshow("weighted", show_frame)

        if depth_frame is not None:
            colored_depth_frame = dispay_colored_depth(depth_frame, "depth")
            cv2.imshow("depth", colored_depth_frame)

        if cv2.waitKey(1) == ord('q'):
            break
