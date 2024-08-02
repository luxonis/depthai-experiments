#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import datetime

DETECTION_ROI = (200,300,1000,700) # Specific to `depth-person-counting-01` recording
path = Path('depth-people-counting-01').resolve().absolute()


class TextHelper: # not important
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

text = TextHelper()

class TestPassthrough(dai.node.ThreadedHostNode): # not important
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()
        self.timestamp = 0
        self.instanceNum = None
        self.frameInterval = 33


    def run(self):
        while self.isRunning():
            buffer : dai.ImgFrame = self.input.get()

            buffer.setInstanceNum(self.instanceNum)
            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)
            buffer.setTimestamp(tstamp)
            buffer.setTimestampDevice(tstamp)
            
            self.output.send(buffer)
            self.timestamp += self.frameInterval


    def setInstanceNum(self, instanceNum):
        self.instanceNum = instanceNum


class InputsConnector(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()

    def run(self):
        while self.isRunning():
            buffer = self.input.get()
            self.output.send(buffer)


def get_detections(contours): # not important
    dets = dai.ImgDetections()
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
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
    return dets


def get_contours(depthFrame): # not important
    cropped = depthFrame[DETECTION_ROI[1]:DETECTION_ROI[3], DETECTION_ROI[0]:DETECTION_ROI[2]]
    ret, thresh = cv2.threshold(cropped, 125, 145, cv2.THRESH_BINARY)
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37,37)))
    edged = cv2.Canny(blob, 20, 80)
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    return contours


def to_planar(arr: np.ndarray) -> list: # not important
    return arr.transpose(2, 0, 1).flatten()


with dai.Pipeline() as pipeline:

    left = pipeline.create(dai.node.ReplayVideo)
    left.setReplayVideoFile(path / 'left.mp4')
    left.setOutFrameType(dai.ImgFrame.Type.RAW8)
    left.setSize(1280, 800)

    right = pipeline.create(dai.node.ReplayVideo)
    right.setReplayVideoFile(path / 'right.mp4')
    right.setOutFrameType(dai.ImgFrame.Type.RAW8)
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
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)


    ### problem ###
    connect_node = pipeline.create(InputsConnector)

    connect_node.output.link(objectTracker.inputDetectionFrame)
    # connect_node.output.link(objectTracker.inputTrackerFrame)
    
    frameInQ = connect_node.input.createInputQueue()
    ### problem ###

    detInQ = objectTracker.inputDetections.createInputQueue()

    depth_q_out = stereo.disparity.createOutputQueue()
    tracklets_out = objectTracker.out.createOutputQueue()

    print("pipeline created")
    pipeline.start()

    while pipeline.isRunning():
        depthFrame = depth_q_out.get().getFrame()
        depthFrame = (depthFrame*(255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        trackletsIn = tracklets_out.tryGet()
        if trackletsIn is not None:
            pass
        else:
            print('no new tracklets')

        contours = get_contours(depthFrame)
        dets = get_detections(contours)   

        detInQ.send(dets)

        ### problem ###
        imgFrame = dai.ImgFrame()
        imgFrame.setType(dai.ImgFrame.Type.BGR888p)
        imgFrame.setFrame(to_planar(depthRgb))
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        frameInQ.send(imgFrame)
        ### problem ###

        text.rectangle(depthRgb, (DETECTION_ROI[0], DETECTION_ROI[1]), (DETECTION_ROI[2], DETECTION_ROI[3]))
        cv2.imshow('depth', depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print("pipeline finished")
