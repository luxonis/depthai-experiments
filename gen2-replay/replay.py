#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import sys
import numpy as np

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--depth', action='store_true', default=False, help="Use saved depth maps")
args = parser.parse_args()

def create_pipeline(mono, depth):
    nodes = {}
    if mono and depth:
        mono = False # Use depth stream

    pipeline = dai.Pipeline()

    nodes['color_in'] = pipeline.createXLinkIn()
    nodes['color_in'].setStreamName("color_in")

    if mono:
        nodes['left_in'] = pipeline.createXLinkIn()
        nodes['left_in'].setStreamName("left_in")
        nodes['right_in'] = pipeline.createXLinkIn()
        nodes['right_in'].setStreamName("right_in")

        nodes['stereo'] = pipeline.createStereoDepth()
        nodes['stereo'].initialConfig.setConfidenceThreshold(240)
        nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        nodes['stereo'].setRectification(False)
        nodes['stereo'].setLeftRightCheck(False)
        nodes['stereo'].setExtendedDisparity(False)
        nodes['stereo'].setSubpixel(False)
        nodes['stereo'].setInputResolution(640,400)

        nodes['left_in'].out.link(nodes['stereo'].left)
        nodes['right_in'].out.link(nodes['stereo'].right)

        right_s_out = pipeline.createXLinkOut()
        right_s_out.setStreamName("rightS")
        nodes['stereo'].syncedRight.link(right_s_out.input)

        left_s_out = pipeline.createXLinkOut()
        left_s_out.setStreamName("leftS")
        nodes['stereo'].syncedLeft.link(left_s_out.input)

    if depth or mono:
        nn = pipeline.createMobileNetSpatialDetectionNetwork()
        nn.setBoundingBoxScaleFactor(0.3)
        nn.setDepthLowerThreshold(100)
        nn.setDepthUpperThreshold(5000)

        if depth:
            nodes['depth_in'] = pipeline.createXLinkIn()
            nodes['depth_in'].setStreamName("depth_in")
            nodes['depth_in'].out.link(nn.inputDepth)
        else:
            nodes['stereo'].depth.link(nn.inputDepth)
    else:
        nn = pipeline.createMobileNetDetectonNetwork()

    nn.setBlobPath("models/mobilenet-ssd_openvino_2021.4_6shave.blob")
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)

    nodes['color_in'].out.link(nn.input)

    bbOut = pipeline.createXLinkOut()
    bbOut.setStreamName("bb_out")
    nn.boundingBoxMapping.link(bbOut.input)

    detOut = pipeline.createXLinkOut()
    detOut.setStreamName("det_out")
    nn.out.link(detOut.input)

    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth_out")
    nn.passthroughDepth.link(depthOut.input)

    rgbOut = pipeline.createXLinkOut()
    rgbOut.setStreamName("nn_passthrough_out")
    nn.passthrough.link(rgbOut.input)
    return pipeline, nodes

class Replay:
    def __init__(self, path):
        self.path = path
        # get all mjpegs
        #  = cv2.VideoCapture(str(Path(path).resolve().absolute()))
        self.cap = {}
        recordings = os.listdir(path)
        if "left.mjpeg" in recordings and "right.mjpeg" in recordings:
            self.cap['left'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'left.mjpeg'))
            self.cap['right'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'right.mjpeg'))
        if "color.mjpeg" in recordings:
            self.cap['color'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'color.mjpeg'))

        if 'left' in self.cap:
            create_pipeline(mono=True, depth=False)
        else:
            create_pipeline(mono=False, depth=False)
        if len(self.cap) == 0:
            raise RuntimeError("There are no .mjpeg recordings in the folder specified.")

    # def __del__(self):
    #     for vid in self.cap:
    #         vid.release()
    def set_queues(self, queues):
        self.queues = queues

    def to_planar(self, arr, shape = None):
        if shape is None: return arr.transpose(2, 0, 1).flatten()
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def get_frames(self):
        frames = {}
        for name in self.cap:
            if not self.cap[name].isOpened(): return None
            ok, frame = self.cap[name].read()
            if ok:
                # cv2.imshow('og_' + name, frame)
                frames[name] = frame
        if len(frames) == 0: return None
        return frames

    def send_frames(self):
        frames = self.get_frames()
        if frames is None: return False # end of recording

        for name in frames:
            self.send_frame(frames[name], name)

        return True

    def send_frame(self, frame, name):
        q_name = name + '_in'
        if q_name in self.queues:
            if name == 'color':
                self.send_color(self.queues[q_name], frame)
            elif name == 'left':
                self.send_mono(self.queues[q_name], frame, False)
            elif name == 'right':
                self.send_mono(self.queues[q_name], frame, True)
            elif name == 'depth':
                self.send_depth(self.queues[q_name], frame)

    def send_mono(self, q, img, right):
        img = img[:,:,0] # all 3 planes are the same
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(cv2.flip(img, 1)) # Flip the rectified frame
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if right else 1))
        q.send(frame)

    def send_color(self, q, img):
        preview = img[0:1080, 420:1500] # Crop before sending
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(preview, (300, 300)))
        frame.setWidth(300)
        frame.setHeight(300)
        frame.setInstanceNum(0)
        q.send(frame)

    def send_depth(self, q, depth):
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
        q.send(frame)

# Pipeline defined, now the device is connected to
replay = Replay(args.path)
mono = 'left' in replay.cap
depth = 'depth' in replay.cap
pipeline, nodes = create_pipeline(mono, depth)
with dai.Device(pipeline) as device:
    queues = {}
    if depth:
        queues['depth_in'] = device.getInputQueue('depth_in')
    elif mono:
        # Use mono frames
        queues['left_in'] = device.getInputQueue('left_in')
        queues['right_in'] = device.getInputQueue('right_in')
        queues['leftS'] = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
        queues['rightS'] = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)

    if 'color' in replay.cap:
        queues['color_in'] = device.getInputQueue('color_in')

    queues['depth_out'] = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)

    queues['bb_out'] = device.getOutputQueue(name="bb_out", maxSize=4, blocking=False)
    queues['det_out'] = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)
    queues['nn_passthrough_out'] = device.getOutputQueue(name="nn_passthrough_out", maxSize=4, blocking=False)

    replay.set_queues(queues)

    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.send_frames():
        inRgb = queues['nn_passthrough_out'].get()
        rgbFrame = inRgb.getCvFrame().reshape((300, 300, 3))

        if mono:
            cv2.imshow("left", queues['leftS'].get().getCvFrame())
            cv2.imshow("right", queues['rightS'].get().getCvFrame())

        depthFrame = queues['depth_out'].get().getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        height = inRgb.getHeight()
        width = inRgb.getWidth()

        inDet = queues['det_out'].tryGet()
        if inDet is not None:
            if len(inDet.detections) != 0:
                # Display boundingbox mappings on the depth frame
                bbMapping = queues['bb_out'].get()
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
    print('End of the recording')

