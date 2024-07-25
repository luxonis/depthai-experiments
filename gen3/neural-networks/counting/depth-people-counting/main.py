#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
from pathlib import Path
import datetime

DETECTION_ROI = (200,300,1000,700) # Specific to `depth-person-counting-01` recording

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
    def rectangle(self, frame, topLeft,bottomRight, size=1.):
        cv2.rectangle(frame, topLeft, bottomRight, self.bg_color, int(size*4))
        cv2.rectangle(frame, topLeft, bottomRight, self.color, int(size))
        return frame

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='depth-people-counting-01', type=str, help="Path to depthai-recording")
args = parser.parse_args()

THRESH_DIST_DELTA = 0.5
class PeopleCounter:
    def __init__(self):
        self.tracking = {}
        self.lost_cnt = {}
        self.people_counter = [0,0,0,0] # Up, Down, Left, Right

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



class Display(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, disp_out : dai.Node.Output, tracklets : dai.Node.Output, detInQ, frameInQ):
        self.text = TextHelper()
        self.counter = PeopleCounter()
        self.detInQ = detInQ
        self.frameInQ = frameInQ

        self.link_args(disp_out, tracklets)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, disp, tracklets):
        depthFrame = disp.getFrame()
        depthFrame = (depthFrame*self.disparity_multiplier).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)  

        if tracklets is not None:
            self.counter.new_tracklets(tracklets.tracklets)

        # Crop only the corridor:
        
        cropped = depthFrame[DETECTION_ROI[1]:DETECTION_ROI[3], DETECTION_ROI[0]:DETECTION_ROI[2]]
        cv2.imshow('Crop', cropped)

        ret, thresh = cv2.threshold(cropped, 125, 145, cv2.THRESH_BINARY)

        blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37,37)))
        # cv2.imshow('blob', blob)

        edged = cv2.Canny(blob, 20, 80)
        # cv2.imshow('Canny', edged)

        contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # cv2.imshow('Rect', text.rectangle(blob, (x,y), (x+w, y+h)))
            x += DETECTION_ROI[0]
            y += DETECTION_ROI[1]
            area = w*h

            if 15000 < area:
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
                self.text.rectangle(depthRgb, (x, y), (x+w, y+h), size=2.5)

        self.detInQ.send(dets)
        imgFrame = dai.ImgFrame()
        imgFrame.setData(self.to_planar(depthRgb))
        imgFrame.setType(dai.ImgFrame.Type.BGR888p)
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        self.frameInQ.send(imgFrame)

        self.text.rectangle(depthRgb, (DETECTION_ROI[0], DETECTION_ROI[1]), (DETECTION_ROI[2], DETECTION_ROI[3]))
        self.text.putText(depthRgb, str(self.counter), (20, 40))

        cv2.imshow('depth', depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.stopPipeline()
        # elif key == 32: # Space
        #     self.togglePause()
    

    def set_disparity_multiplier(self, disp_multiplier):
        self.disparity_multiplier = disp_multiplier

    def to_planar(self, arr: np.ndarray) -> list:
        return arr.transpose(2, 0, 1).flatten()


frameInterval = 33
class TestPassthrough(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()
        self.timestamp = 0
        self.instanceNum = None


    def run(self):
        while self.isRunning():
            buffer : dai.ImgFrame = self.input.get()

            buffer.setInstanceNum(self.instanceNum)
            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)
            buffer.setTimestamp(tstamp)
            buffer.setTimestampDevice(tstamp)
            buffer.setType(dai.ImgFrame.Type.GRAY8)
            
            self.output.send(buffer)
            self.timestamp += frameInterval


    def setInstanceNum(self, instanceNum):
        self.instanceNum = instanceNum


def to_planar(arr: np.ndarray) -> list:
    return arr.transpose(2, 0, 1).flatten()


with dai.Pipeline() as pipeline:

    path = Path(args.path).resolve().absolute()

    left = pipeline.create(dai.node.ReplayVideo)
    left.setReplayVideoFile(path / 'left.mp4')
    left.setOutFrameType(dai.ImgFrame.Type.GRAY8) # RAW8
    left.setSize(1280, 800)

    right = pipeline.create(dai.node.ReplayVideo)
    right.setReplayVideoFile(path / 'right.mp4')
    right.setOutFrameType(dai.ImgFrame.Type.GRAY8) # RAW8
    right.setSize(1280, 800)
    
    pipeline.setCalibrationData(dai.CalibrationHandler(str(path / 'calib.json')))

    host1 = pipeline.create(TestPassthrough)
    host1.setInstanceNum(dai.CameraBoardSocket.CAM_B)
    host2 = pipeline.create(TestPassthrough)
    host2.setInstanceNum(dai.CameraBoardSocket.CAM_C)

    left.out.link(host1.input)
    right.out.link(host2.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=host1.output, right=host2.output)

    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7) # KERNEL_7x7 default
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    objectTracker = pipeline.create(dai.node.ObjectTracker)
    objectTracker.inputTrackerFrame.setBlocking(True)
    objectTracker.inputDetectionFrame.setBlocking(True)
    objectTracker.inputDetections.setBlocking(True)
    objectTracker.setDetectionLabelsToTrack([1])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    print("pipeline created")

    detInQ = objectTracker.inputDetections.createInputQueue()
    frameInQ = objectTracker.inputDetectionFrame.createInputQueue()

    tracklets_out = objectTracker.out.createOutputQueue()
    depth_q_out = stereo.disparity.createOutputQueue()

    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()
    text = TextHelper()
    counter = PeopleCounter()


    pipeline.start()

    while pipeline.isRunning():
        depthFrame = depth_q_out.get().getFrame()
        depthFrame = (depthFrame*disparityMultiplier).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        trackletsIn = tracklets_out.tryGet()
        if trackletsIn is not None:
            counter.new_tracklets(trackletsIn.tracklets)
            print("new tracklets")

        # Crop only the corridor:
        
        cropped = depthFrame[DETECTION_ROI[1]:DETECTION_ROI[3], DETECTION_ROI[0]:DETECTION_ROI[2]]
        cv2.imshow('Crop', cropped)

        ret, thresh = cv2.threshold(cropped, 125, 145, cv2.THRESH_BINARY)

        blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37,37)))
        # cv2.imshow('blob', blob)

        edged = cv2.Canny(blob, 20, 80)
        # cv2.imshow('Canny', edged)

        contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # cv2.imshow('Rect', text.rectangle(blob, (x,y), (x+w, y+h)))
            x += DETECTION_ROI[0]
            y += DETECTION_ROI[1]
            area = w*h

            if 15000 < area:
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
                text.rectangle(depthRgb, (x, y), (x+w, y+h), size=2.5)

        detInQ.send(dets)
        imgFrame = dai.ImgFrame()
        imgFrame.setFrame(to_planar(depthRgb))
        imgFrame.setType(dai.ImgFrame.Type.BGR888p)
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        frameInQ.send(imgFrame)

        text.rectangle(depthRgb, (DETECTION_ROI[0], DETECTION_ROI[1]), (DETECTION_ROI[2], DETECTION_ROI[3]))
        text.putText(depthRgb, str(counter), (20, 40))

        cv2.imshow('depth', depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    print("pipeline finished")

    # objectTracker.inputDetectionFrame .. frameIn
    # objectTracker.inputDetections .. detIn

    # stereo.disparity .. depthOut
    # objectTracker.out .. trackletsOut
