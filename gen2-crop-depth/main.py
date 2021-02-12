#!/usr/bin/env python3
import csv
import queue
import threading
import time
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np

# Get argument first
mobilenet_path = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)
stereo.setOutputRectified(True) # The rectified streams are horizontally mirrored by default
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)
cam_rgb.preview.link(detection_nn.input)

manipDepth = pipeline.createImageManip()
manipDepth.setWaitForConfigInput(True)
manipDepth.setMaxOutputFrameSize(3000000)
stereo.disparity.link(manipDepth.inputImage)

# Create outputs
xin_conf = pipeline.createXLinkIn()
xin_conf.setStreamName("manipconf")
xin_conf.out.link(manipDepth.inputConfig)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.disparity.link(xout_depth.input)

xout_manip = pipeline.createXLinkOut()
xout_manip.setStreamName("manip")
manipDepth.out.link(xout_manip.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# MobilenetSSD label texts
texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

for text in texts:
    (Path(__file__).parent / Path(f'data/{text}')).mkdir(parents=True, exist_ok=True)
(Path(__file__).parent / Path(f'data/raw')).mkdir(parents=True, exist_ok=True)


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device, open('data/dataset.csv', 'w') as dataset_file:
    dataset = csv.DictWriter(
        dataset_file,
        ["timestamp", "label", "left", "top", "right", "bottom", "raw_frame", "overlay_frame", "cropped_frame"]
    )
    dataset.writeheader()

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_conf = device.getInputQueue(name="manipconf")

    frame = None
    frame_depth = None
    frame_manip = None
    parse_bbox = None
    awaiting_depth = False
    bboxes = []
    labels = []
    crop_q = queue.Queue()


    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_rgb is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
            frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if in_nn is not None:
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # transform the 1D array into Nx7 matrix
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # filter out the results which confidence less than a defined threshold
            bboxes = bboxes[bboxes[:, 2] > 0.5]
            # Cut bboxes and labels
            labels = bboxes[:, 1].astype(int)
            bboxes = bboxes[:, 3:7]
            for bbox in bboxes:
                crop_q.put(bbox)

        if in_depth is not None:
            frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
            frame_depth = np.ascontiguousarray(frame_depth)
            frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

        if not crop_q.empty():
            parse_bbox = crop_q.get()
            new_conf = dai.ImageManipConfig()
            new_conf.setCropRect(*parse_bbox)
            q_conf.send(new_conf)
            q_manip.tryGetAll()

        if len(bboxes) > 0:
            in_manip = q_manip.tryGet()
            if in_manip is not None:
                frame_manip = in_manip.getData().reshape((in_manip.getHeight(), in_manip.getWidth())).astype(np.uint8)
                frame_manip = np.ascontiguousarray(frame_manip)
                frame_manip = cv2.applyColorMap(frame_manip, cv2.COLORMAP_JET)
        else:
            frame_manip = None

        if frame_manip is not None:
            cv2.imshow("disparity-roi", frame_manip)

        if frame_depth is not None:
            for raw_bbox, label in zip(bboxes, labels):
                bbox = frame_norm(frame_depth, raw_bbox)
                cv2.rectangle(frame_depth, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame_depth, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("disparity", frame_depth)

        if frame is not None:
            debug_frame = frame.copy()
            # if the frame is available, draw bounding boxes on it and show the frame
            for raw_bbox, label in zip(bboxes, labels):
                bbox = frame_norm(debug_frame, raw_bbox)
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(debug_frame, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("rgb", debug_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    if thread is not None:
        thread.join()
