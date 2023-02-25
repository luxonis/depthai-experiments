#!/usr/bin/env python3
import argparse

import cv2
import depthai as dai
import numpy as np

from depthai_sdk import Replay

DETECTION_ROI = (200, 300, 1000, 700)  # Specific to `depth-person-counting-01` recording


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.3, self.bg_color, 5, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.3, self.color, 2, self.line_type)
        return frame

    def rectangle(self, frame, topLeft, bottomRight, size=1.):
        cv2.rectangle(frame, topLeft, bottomRight, self.bg_color, int(size * 4))
        cv2.rectangle(frame, topLeft, bottomRight, self.color, int(size))
        return frame


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='depth-people-counting-01', type=str, help="Path to depthai-recording")
args = parser.parse_args()


def to_planar(arr: np.ndarray) -> list:
    return arr.transpose(2, 0, 1).flatten()


THRESH_DIST_DELTA = 0.5


class PeopleCounter:
    def __init__(self):
        self.tracking = {}
        self.lost_cnt = {}
        self.people_counter = [0, 0, 0, 0]  # Up, Down, Left, Right

    def __str__(self) -> str:
        return f"Left: {self.people_counter[2]}, Right: {self.people_counter[3]}"

    def tracklet_removed(self, coords1, coords2):
        deltaX = coords2[0] - coords1[0]
        # print('Delta X', deltaX)

        if THRESH_DIST_DELTA < abs(deltaX):
            self.people_counter[2 if 0 > deltaX else 3] += 1
            direction = "left" if 0 > deltaX else "right"
            print(f"Person moved {direction}")

    def get_centroid(self, roi):
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y
        return ((x2 + x1) / 2, (y2 + y1) / 2)

    def new_tracklets(self, tracklets):
        for t in tracklets:
            # If new tracklet, save its centroid
            if t.status == dai.Tracklet.TrackingStatus.NEW:
                self.tracking[str(t.id)] = self.get_centroid(t.roi)
                self.lost_cnt[str(t.id)] = 0
            elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.lost_cnt[str(t.id)] = 0
            elif t.status == dai.Tracklet.TrackingStatus.LOST:
                self.lost_cnt[str(t.id)] += 1
                # Tracklet has been lost for too long
                if 10 < self.lost_cnt[str(t.id)]:
                    self.lost_cnt[str(t.id)] = -999
                    self.tracklet_removed(self.tracking[str(t.id)], self.get_centroid(t.roi))
            elif t.status == dai.Tracklet.TrackingStatus.REMOVED:
                if 0 <= self.lost_cnt[str(t.id)]:
                    self.lost_cnt[str(t.id)] = -999
                    self.tracklet_removed(self.tracking[str(t.id)], self.get_centroid(t.roi))


# Create Replay object
replay = Replay(args.path)

# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.initPipeline()

nodes.stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7) # KERNEL_7x7 default
nodes.stereo.setLeftRightCheck(True)
# nodes.stereo.setSubpixel(True)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depthOut")
nodes.stereo.disparity.link(depthOut.input)

objectTracker = pipeline.createObjectTracker()
objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
objectTracker.setDetectionLabelsToTrack([1])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

# Linking
xinFrame = pipeline.createXLinkIn()
xinFrame.setStreamName("frameIn")
xinFrame.out.link(objectTracker.inputDetectionFrame)

# Maybe we need to send the old frame here, not sure
xinFrame.out.link(objectTracker.inputTrackerFrame)

xinDet = pipeline.createXLinkIn()
xinDet.setStreamName("detIn")
xinDet.out.link(objectTracker.inputDetections)

trackletsOut = pipeline.createXLinkOut()
trackletsOut.setStreamName("trackletsOut")
objectTracker.out.link(trackletsOut.input)

with dai.Device(pipeline) as device:
    replay.createQueues(device)

    depthQ = device.getOutputQueue(name="depthOut", maxSize=4, blocking=False)
    trackletsQ = device.getOutputQueue(name="trackletsOut", maxSize=4, blocking=False)

    detInQ = device.getInputQueue("detIn")
    frameInQ = device.getInputQueue("frameIn")

    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()

    text = TextHelper()
    counter = PeopleCounter()

    while replay.sendFrames():
        depthFrame = depthQ.get().getFrame()
        depthFrame = (depthFrame * disparityMultiplier).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        trackletsIn = trackletsQ.tryGet()
        if trackletsIn is not None:
            counter.new_tracklets(trackletsIn.tracklets)

        # Crop only the corridor:

        cropped = depthFrame[DETECTION_ROI[1]:DETECTION_ROI[3], DETECTION_ROI[0]:DETECTION_ROI[2]]
        cv2.imshow('Crop', cropped)

        ret, thresh = cv2.threshold(cropped, 125, 145, cv2.THRESH_BINARY)

        blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
        # cv2.imshow('blob', blob)

        edged = cv2.Canny(blob, 20, 80)
        # cv2.imshow('Canny', edged)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # cv2.imshow('Rect', text.rectangle(blob, (x,y), (x+w, y+h)))
            x += DETECTION_ROI[0]
            y += DETECTION_ROI[1]
            area = w * h

            if 15000 < area:
                # Send the detection to the device - ObjectTracker node
                det = dai.ImgDetection()
                det.label = 1
                det.confidence = 1.0
                det.xmin = x
                det.ymin = y
                det.xmax = x + w
                det.ymax = y + h
                dets.detections = [det]

                # Draw rectangle on the biggest countour
                text.rectangle(depthRgb, (x, y), (x + w, y + h), size=2.5)

        detInQ.send(dets)
        imgFrame = dai.ImgFrame()
        imgFrame.setData(to_planar(depthRgb))
        imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        frameInQ.send(imgFrame)

        text.rectangle(depthRgb, (DETECTION_ROI[0], DETECTION_ROI[1]), (DETECTION_ROI[2], DETECTION_ROI[3]))
        text.putText(depthRgb, str(counter), (20, 40))

        cv2.imshow('depth', depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:  # Space
            replay.togglePause()

    print('End of the recording')
