#!/usr/bin/env python3
import argparse
import depthai as dai
import numpy as np
from depthai_sdk import Replay
from calc import HostSpatialsCalc
import math
import matplotlib.pyplot as plt
import cv2

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)
    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 3)
        cv2.rectangle(frame, p1, p2, self.color, 1)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay object
replay = Replay(args.path)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline = replay.initPipeline()

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
replay.stereo.depth.link(depthOut.input)

right_s_out = pipeline.create(dai.node.XLinkOut)
right_s_out.setStreamName("rightS")
replay.stereo.syncedRight.link(right_s_out.input)

left_s_out = pipeline.create(dai.node.XLinkOut)
left_s_out.setStreamName("leftS")
replay.stereo.syncedLeft.link(left_s_out.input)

replay.stereo.initialConfig.setConfidenceThreshold(190)
replay.stereo.setLeftRightCheck(True)
replay.stereo.setSubpixel(True)
# nodes.stereo.useHomographyRectification(False)

with dai.Device(pipeline) as device:
    replay.createQueues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    rightS_Q = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)
    leftS_Q = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)

    disparityMultiplier = 255 / replay.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)

    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y = 350
    x = 650
    step = 3
    delta = 3
    hostSpatials.setDeltaRoi(delta)

    cv2.namedWindow("depth")
    points = None
    def cb(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONUP:
            if points == (x, y):
                points = None # Clear
            else:
                points = (x, y)
                print("new points", points)
    cv2.setMouseCallback("depth", cb)

    pause = False
    i = 0
    depthFrame = None
    rightFrame = None
    spatials = None
    started = False

    z_arr = []

    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.sendFrames(pause):
        i += 1
        if i == 40:
            pause = True

        # if mono:
        if leftS_Q.has():
            leftFrame = leftS_Q.get().getCvFrame()
        if rightS_Q.has():
            rightFrame = rightS_Q.get().getCvFrame()

        if depthQ.has():
            depthFrame = depthQ.get().getFrame()
            if 40 < i and started:
                print(spatials['z'])
                z_arr.append(spatials["z"])

        if depthFrame is not None:
            spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x, y))

            depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
            # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

            if points is not None:
                t = "{}mm".format(depthFrame[points[1]][points[0]])
                cv2.circle(depthFrameColor, points, 3, (255, 255, 255), -1)
                cv2.putText(depthFrameColor, t, (points[0] + 5, points[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))

            text.rectangle(depthFrameColor, (x - delta, y - delta), (x + delta, y + delta))

            text.putText(depthFrameColor, "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                        (x + 10, y + 20))
            text.putText(depthFrameColor, "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                        (x + 10, y + 35))
            text.putText(depthFrameColor, "Z: " + ("{:.3f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                        (x + 10, y + 50))
            # Show the frame
            cv2.imshow("depth", depthFrameColor)
        if rightFrame is not None:
            rc = rightFrame.copy()
            text.rectangle(rc, (x - delta, y - delta), (x + delta, y + delta))
            cv2.imshow("right", rc)

            cv2.imshow("left", leftFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            y -= step
        elif key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('d'):
            x += step
        elif key == ord('r'):  # Increase Delta
            if delta < 50:
                delta += 5
                hostSpatials.setDeltaRoi(delta)
        elif key == ord('f'):  # Decrease Delta
            if 3 < delta:
                delta -= 5
                hostSpatials.setDeltaRoi(delta)
        elif key == ord(' '):
            started = True
            pause = not pause
    print('End of the recording')

    plt.hist(z_arr, bins=33)
    plt.show()