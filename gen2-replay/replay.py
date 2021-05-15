#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import sys
import numpy as np
import time

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--depth', action='store_true', default=False, help="Use saved depth maps")
args = parser.parse_args()

# Get the stored frames path
dest = Path(args.path).resolve().absolute()
frames = os.listdir(str(dest))
frames_sorted = sorted([int(i) for i in frames])

pipeline = dai.Pipeline()

rgb_in = pipeline.createXLinkIn()
rgb_in.setStreamName("rgbIn")

if not args.depth:
    left_in = pipeline.createXLinkIn()
    right_in = pipeline.createXLinkIn()
    left_in.setStreamName("left")
    right_in.setStreamName("right")

    stereo = pipeline.createStereoDepth()
    stereo.setConfidenceThreshold(240)
    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    stereo.setMedianFilter(median)
    # stereo.setEmptyCalibration()
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    stereo.setInputResolution(640,400)

    left_in.out.link(stereo.left)
    right_in.out.link(stereo.right)

    right_s_out = pipeline.createXLinkOut()
    right_s_out.setStreamName("rightS")
    stereo.syncedRight.link(right_s_out.input)

    left_s_out = pipeline.createXLinkOut()
    left_s_out.setStreamName("leftS")
    stereo.syncedLeft.link(left_s_out.input)

spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
spatialDetectionNetwork.setBlobPath("models/mobilenet-ssd_openvino_2021.2_6shave.blob")
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.3)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

if args.depth:
    depth_in = pipeline.createXLinkIn()
    depth_in.setStreamName("depthIn")
    depth_in.out.link(spatialDetectionNetwork.inputDepth)
else:
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

rgb_in.out.link(spatialDetectionNetwork.input)

bbOut = pipeline.createXLinkOut()
bbOut.setStreamName("bb")
spatialDetectionNetwork.boundingBoxMapping.link(bbOut.input)

detOut = pipeline.createXLinkOut()
detOut.setStreamName("det")
spatialDetectionNetwork.out.link(detOut.input)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depth")
spatialDetectionNetwork.passthroughDepth.link(depthOut.input)

rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("rgb")
spatialDetectionNetwork.passthrough.link(rgbOut.input)

class Replay:
    def __init__(self, path, device):
        self.path = path

        # Create input queues
        inputs = ["rgbIn"]
        if args.depth:
            inputs.append("depthIn")
        else: # Use mono frames
            inputs.append("left")
            inputs.append("right")
        self.q = {}
        for input_name in inputs:
            self.q[input_name] = device.getInputQueue(input_name)
        print(self.q.items())
        print(type(device))
        print(device.getMxId())

    def to_planar(self, arr, shape):
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def read_files(self, frame_folder, files):
        frames = {}
        for file in files:
            file_path = str((Path(self.path) / str(frame_folder) / file).resolve().absolute())
            if file.startswith("right") or file.startswith("left"):
                frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            elif file.startswith("depth"):
                with open(file_path, "rb") as depth_file:
                    frame = list(depth_file.read())
            elif file.startswith("color"):
                frame = cv2.imread(file_path)
            else: print(f"Unknown file found! {file}")
            frames[os.path.splitext(file)[0]] = frame
        return frames

    def get_files(self, frame_folder):
        return os.listdir(str((Path(self.path) / str(frame_folder)).resolve().absolute()))

    def send_frames(self, images):
        for name, img in images.items():
            replay.send_frame(name, img)

    # Send recorded frames from the host to the depthai
    def send_frame(self, name, frame):
        print("sending frame", name)
        if name in ["left", "right"] and not args.depth:
            self.send_mono(frame, name)
        elif name == "depth" and args.depth:
            self.send_depth(frame)
        elif name == "color":
            self.send_rgb(frame)

    def send_mono(self, img, name):
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(cv2.flip(img, 1)) # Flip the rectified frame
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if name == "right" else 1))
        self.q[name].send(frame)
    def send_rgb(self, img):
        preview = img[0:1080, 420:1500] # Crop before sending
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(preview, (300, 300)))
        frame.setWidth(300)
        frame.setHeight(300)
        frame.setInstanceNum(0)
        self.q["rgbIn"].send(frame)
    def send_depth(self, depth):
        # print("depth size", type(depth))
        # depth_frame = np.array(depth).astype(np.uint8).view(np.uint16).reshape((400, 640))
        # depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        # cv2.imshow("depth", depthFrameColor)
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        self.q["depthIn"].send(frame)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    device.startPipeline()

    if not args.depth:
        qLeftS = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
        qRightS = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)

    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    qBb = device.getOutputQueue(name="bb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    qRgbOut = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    replay = Replay(path=args.path, device=device)
    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    for frame_folder in frames_sorted:
        files = replay.get_files(frame_folder)

        # Read the frames from the FS
        images = replay.read_files(frame_folder, files)

        replay.send_frames(images)
        # Send first frames twice for first iteration (depthai FW limitation)
        if frame_folder == 0: replay.send_frames(images)

        inRgb = qRgbOut.get()
        print("ASD")
        rgbFrame = inRgb.getCvFrame().reshape((300, 300, 3))

        if not args.depth:
            cv2.imshow("left", qLeftS.get().getCvFrame())
            cv2.imshow("right", qRightS.get().getCvFrame())

        depthFrame = qDepth.get().getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        height = inRgb.getHeight()
        width = inRgb.getWidth()

        inDet = qDet.tryGet()
        if inDet is not None:
            if len(inDet.detections) != 0:
                # Display boundingbox mappings on the depth frame
                bbMapping = qBb.get()
                roiDatas = bbMapping.getConfigData()
                for roiData in roiDatas:
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0,255,0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Display (spatial) object detections on the color frame
            for detection in inDet.detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * 300)
                x2 = int(detection.xmax * 300)
                y1 = int(detection.ymin * 300)
                y2 = int(detection.ymax * 300)
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(rgbFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break

