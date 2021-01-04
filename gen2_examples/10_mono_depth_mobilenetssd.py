#!/usr/bin/env python3

from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setCamId(1)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setCamId(2)

# Create a node that will produce the depth map
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True) # The rectified streams are horizontally mirrored by default
depth.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
left.out.link(depth.left)
right.out.link(depth.right)

# Create a node to convert the grayscale frame into the nn-acceptable form
manip = pipeline.createImageManip()
manip.setResize(300, 300)
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
depth.rectifiedLeft.link(manip.inputImage)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/mobilenet-ssd.blob')).resolve().absolute()))
manip.out.link(detection_nn.input)

# Create outputs
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.disparity.link(xout_depth.input)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
manip.out.link(xout_left.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
q_left = device.getOutputQueue("left")
q_depth = device.getOutputQueue("depth")
q_nn = device.getOutputQueue("nn")

frame_left = None
frame_depth = None
bboxes = []


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_left = q_left.tryGet()
    in_nn = q_nn.tryGet()
    in_depth = q_depth.tryGet()

    if in_left is not None:
        # if the grayscale frame data is available, transform the 1D data into a HxWxC frame
        shape = (3, in_left.getHeight(), in_left.getWidth())
        frame_left = in_left.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame_left = np.ascontiguousarray(frame_left)

    if in_nn is not None:
        # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # take only the results before -1 digit
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # transform the 1D array into Nx7 matrix
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # filter out the results which confidence less than a defined threshold
        bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]

    if in_depth is not None:
        # data is originally represented as a flat 1D array, it needs to be converted into HxW form
        frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
        frame_depth = np.ascontiguousarray(frame_depth)
        # frame is transformed, the color map will be applied to highlight the depth info
        frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

    if frame_left is not None:
        # if the frame is available, draw bounding boxes on it and show the frame
        for raw_bbox in bboxes:
            bbox = frame_norm(frame_left, raw_bbox)
            cv2.rectangle(frame_left, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("left", frame_left)

    if frame_depth is not None:
        cv2.imshow("depth", frame_depth)

    if cv2.waitKey(1) == ord('q'):
        break
