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


class Replay:
    def __init__(self, path):
        self.path = path

        self.cap = {} # VideoCapture objects
        self.size = {} # Frame sizes

        # steam_types = ['color', 'left', 'right', 'depth']
        # extensions = ['mjpeg', 'avi', 'mp4']

        recordings = os.listdir(path)
        if "left.mjpeg" in recordings and "right.mjpeg" in recordings:
            self.cap['left'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'left.mjpeg'))
            self.cap['right'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'right.mjpeg'))
        if "color.mjpeg" in recordings:
            self.cap['color'] = cv2.VideoCapture(str(Path(path).resolve().absolute() / 'color.mjpeg'))

        if len(self.cap) == 0:
            raise RuntimeError("There are no .mjpeg recordings in the folder specified.")

        # Read basic info about the straems (resolution of streams etc.)
        for name in self.cap:
            self.size[name] = self.get_size(self.cap[name])

        self.color_size = None
        self.keep_ar = False

    # Resize color frames prior to sending them to the device
    def resize_color(self, size):
        self.color_size = size
    def keep_aspect_ratio(self, keep_aspect_ratio):
        self.keep_ar = keep_aspect_ratio

    def init_pipeline(self):
        nodes = {}
        mono = 'left' in self.cap
        depth = 'depth' in self.cap
        if mono and depth:
            mono = False # Use depth stream by default

        pipeline = dai.Pipeline()

        nodes['color'] = pipeline.createXLinkIn()
        nodes['color'].setStreamName("color_in")

        if mono:
            nodes['left'] = pipeline.createXLinkIn()
            nodes['left'].setStreamName("left_in")
            nodes['right'] = pipeline.createXLinkIn()
            nodes['right'].setStreamName("right_in")

            nodes['stereo'] = pipeline.createStereoDepth()
            nodes['stereo'].initialConfig.setConfidenceThreshold(240)
            nodes['stereo'].setRectification(False)
            nodes['stereo'].setInputResolution(self.size['left'])

            nodes['left'].out.link(nodes['stereo'].left)
            nodes['right'].out.link(nodes['stereo'].right)

            right_s_out = pipeline.createXLinkOut()
            right_s_out.setStreamName("rightS")
            nodes['stereo'].syncedRight.link(right_s_out.input)

            left_s_out = pipeline.createXLinkOut()
            left_s_out.setStreamName("leftS")
            nodes['stereo'].syncedLeft.link(left_s_out.input)

        if depth:
            nodes['depth'] = pipeline.createXLinkIn()
            nodes['depth'].setStreamName("depth_in")

        if depth or mono:
            nn = pipeline.createMobileNetSpatialDetectionNetwork()
            nn.setBoundingBoxScaleFactor(0.3)
            nn.setDepthLowerThreshold(100)
            nn.setDepthUpperThreshold(5000)
            

        nn.setBlobPath("models/mobilenet-ssd_openvino_2021.4_6shave.blob")
        nn.setConfidenceThreshold(0.5)
        nn.input.setBlocking(False)

        nodes['color'].out.link(nn.input)


        detOut = pipeline.createXLinkOut()
        detOut.setStreamName("det_out")
        nn.out.link(detOut.input)

        depthOut = pipeline.createXLinkOut()
        depthOut.setStreamName("depth_out")
        # nn.passthroughDepth.link(depthOut.input)
        nodes['stereo'].disparity.link(depthOut.input)

        rgbOut = pipeline.createXLinkOut()
        rgbOut.setStreamName("nn_passthrough_out")
        nn.passthrough.link(rgbOut.input)
        return pipeline, nodes

    def create_queues(self, device):
        self.queues['left_in'] = device.getInputQueue('left_in')
        self.queues['right_in'] = device.getInputQueue('right_in')
        self.queues['color_in'] = device.getInputQueue('color_in')

    def to_planar(self, arr, shape = None):
        if shape is None: return arr.transpose(2, 0, 1).flatten()
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def get_frames(self):
        frames = {}
        for name in self.cap:
            if not self.cap[name].isOpened(): return None
            ok, frame = self.cap[name].read()
            if ok:
                frames[name] = frame
        if len(frames) == 0: return None
        return frames

    def send_frames(self):
        frames = self.get_frames()
        if frames is None: return False # end of recording

        for name in frames:
            self.send_frame(frames[name], name)

        return True

    def get_size(self, cap):
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

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
        frame.setData(img)
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if right else 1))
        q.send(frame)

    def send_color(self, q, img):
        # TODO Use self.color_size to crop & resize
        if self.color_size is not None:
            if self.keep_ar:

                img = cv2.resize(img, self.color_size)
                # Crop to keep desired aspect ratio, resize later
            else: img = cv2.resize(img, self.color_size)

        h, w, c = img.shape
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(img))
        frame.setWidth(w)
        frame.setHeight(h)
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
pipeline, nodes = replay.init_pipeline()

with dai.Device(pipeline) as device:
    queues = {}
    # if depth:
    #     queues['depth_in'] = device.getInputQueue('depth_in')
    # elif mono:
        # Use mono frames
    queues['leftS'] = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
    queues['rightS'] = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)

    if 'color' in replay.cap:
        

    queues['depth_out'] = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)

    queues['bb_out'] = device.getOutputQueue(name="bb_out", maxSize=4, blocking=False)
    queues['det_out'] = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)
    queues['nn_passthrough_out'] = device.getOutputQueue(name="nn_passthrough_out", maxSize=4, blocking=False)

    replay.set_queues(queues)

    disparityMultiplier = 255 / nodes['stereo'].getMaxDisparity()
    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.send_frames():
        inRgb = queues['nn_passthrough_out'].get()
        rgbFrame = inRgb.getCvFrame() #.reshape((300, 300, 3))

        # if mono:
        cv2.imshow("left", queues['leftS'].get().getCvFrame())
        cv2.imshow("right", queues['rightS'].get().getCvFrame())

        depthFrame = queues['depth_out'].get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
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

