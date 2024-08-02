#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import Replay
import argparse

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

def to_planar(arr: np.ndarray) -> list: # not important
    return arr.transpose(2, 0, 1).flatten()

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



### problematic part in v3 ### 
xinFrame = pipeline.createXLinkIn()
xinFrame.setStreamName("frameIn")

xinFrame.out.link(objectTracker.inputDetectionFrame)
xinFrame.out.link(objectTracker.inputTrackerFrame)
### problematic part in v3 ### 

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

    text = TextHelper()

    while replay.sendFrames():
        depthFrame = depthQ.get().getFrame()
        depthFrame = (depthFrame*(255 / nodes.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        depthRgb = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        trackletsIn = trackletsQ.tryGet()
        if trackletsIn is not None:
            pass
        else:
            print("no new")
        
        contours = get_contours(depthFrame)
        dets = get_detections(contours)   

        detInQ.send(dets)

        detInQ.send(dets)
        imgFrame = dai.ImgFrame()
        imgFrame.setData(to_planar(depthRgb))
        imgFrame.setType(dai.RawImgFrame.Type.BGR888p)
        imgFrame.setWidth(depthRgb.shape[0])
        imgFrame.setHeight(depthRgb.shape[1])
        frameInQ.send(imgFrame)
        
        text.rectangle(depthRgb, (DETECTION_ROI[0], DETECTION_ROI[1]), (DETECTION_ROI[2], DETECTION_ROI[3]))
        cv2.imshow('depth', depthRgb)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print('End of the recording')
