#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
import numpy as np
from libraries.depthai_replay import Replay
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="recordings/people-corridor", type=str, help="Path where to store the captured data")
args = parser.parse_args()

def to_planar(arr: np.ndarray) -> list:
    return arr.transpose(2, 0, 1).flatten()

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.3, self.bg_color, 5, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.3, self.color, 2, self.line_type)
    def rectangle(self, frame, topLeft,bottomRight):
        cv2.rectangle(frame, topLeft, bottomRight, self.bg_color, 4)
        cv2.rectangle(frame, topLeft, bottomRight, self.color, 1)


THRESH_DIST_DELTA = 0.5
class PeopleCounter:
    def __init__(self):
        self.frame = None
        self.tracking = {}
        self.lost_cnt = {}
        self.people_counter = 0
    def tracklet_removed(self, coords1, coords2):
        deltaX = coords2[0] - coords1[0]
        # print('Delta X', deltaX)

        if THRESH_DIST_DELTA < abs(deltaX):
            self.people_counter += -1 if 0 > deltaX else 1
            direction = "left" if 0 > deltaX else "right"
            print(f"Person moved {direction}")

    def get_centroid(self, roi):
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y
        return ((x2+x1)/2, (y2+y1)/2)

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
replay.disable_stream('color')

# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.init_pipeline()

nodes['stereo'].initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7) # KERNEL_7x7 default
nodes['stereo'].setLeftRightCheck(True)
nodes['stereo'].setSubpixel(True)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depthOut")
nodes['stereo'].disparity.link(depthOut.input)

objectTracker = pipeline.createObjectTracker()
objectTracker.inputTrackerFrame.setBlocking(True)
objectTracker.inputDetectionFrame.setBlocking(True)
objectTracker.inputDetections.setBlocking(True)
objectTracker.setDetectionLabelsToTrack([1])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

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
    queues = {}
    replay.create_queues(device)

    depthQ = device.getOutputQueue(name="depthOut", maxSize=4, blocking=False)
    trackletsQ = device.getOutputQueue(name="trackletsOut", maxSize=4, blocking=False)

    detInQ = device.getInputQueue("detIn")
    frameInQ = device.getInputQueue("frameIn")

    disparityMultiplier = 255 / 760

    out = cv2.VideoWriter('depthCounting.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,400))
    text = TextHelper()
    counter = PeopleCounter()

    while replay.send_frames():
        depthFrame = depthQ.get().getFrame()
        depthFrame = (depthFrame*disparityMultiplier).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        trackletsIn = trackletsQ.tryGet()
        if trackletsIn is not None:
            counter.new_tracklets(trackletsIn.tracklets)

        # Crop only the corridor:
        cropped = depthFrame[60:270,:500]

        ret, thresh = cv2.threshold(cropped, 45, 55, cv2.THRESH_BINARY)

        blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13)))
        cv2.imshow('blob', blob)

        edged = cv2.Canny(blob, 20, 80)
        # cv2.imshow("edges1", edged)
        contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            y+=60
            area = w*h

            if 10000 < area:
                # Send the detection to the device - ObjectTracker node
                det = dai.ImgDetection()
                det.label = 1
                det.confidence=1.0
                det.xmin = x
                det.ymin = y
                det.xmax = x + w
                det.ymax = y + h
                dets.detections = [det]

                # Draw rectangle on the biggest countour
                text.rectangle(depthRgb, (x, y), (x+w, y+h))

        detInQ.send(dets)
        imgFrame = dai.ImgFrame()
        imgFrame.setData(to_planar(depthRgb))
        imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        frameInQ.send(imgFrame)

        text.putText(depthRgb, f"People inside: {counter.people_counter}", (150, 380))

        cv2.imshow("depth", depthRgb)
        out.write(depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'): # Pause
            print('Press `c` to continute playing the video')
            while True:
                if cv2.waitKey(1) == ord('c'):
                    break
    print('End of the recording')
    out.release()
