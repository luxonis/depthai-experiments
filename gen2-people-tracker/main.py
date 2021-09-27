#!/usr/bin/env python3

import sys
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from time import monotonic

# Labels
labelMap = ["background", "person" ]

# Get argument first
parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help=".blob path")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-spi', '--spi', action='store_true', default=False, help="Send tracklets to the MCU via SPI")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI RGB camera for inference (conflicts with -vid)")
parser.add_argument('-t', '--threshold', default=0.25, type=float,
    help="Minimum distance the person has to move (across the x/y axis) to be considered a real movement")
args = parser.parse_args()

parentDir = Path(__file__).parent

videoPath = args.video or parentDir / Path('demo/example_01.mp4')
nnPath = args.nn or blobconverter.from_zoo(name="person-detection-retail-0013", shaves=7)

# Whether we want to use video from host or rgb camera
VIDEO = not args.camera

# Whether we want to send tracklets via SPI to the MCU
SPI = args.spi

# Minimum distance the person has to move (across the x/y axis) to be considered a real movement
THRESH_DIST_DELTA = args.threshold

# Start defining a pipeline
pipeline = dai.Pipeline()

# Create and configure the detection network
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
detectionNetwork.setBlobPath(str(Path(nnPath).resolve().absolute()))
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)

if VIDEO:
    # Configure XLinkIn - we will send video through it
    videoIn = pipeline.createXLinkIn()
    videoIn.setStreamName("video_in")

    # NOTE: video must be of size 544x320. We will resize this on the
    # host, but you could also use ImageManip node to do it on device

    # Link video in with the detection network
    videoIn.out.link(detectionNetwork.input)

else:
    # Create and configure the color camera
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(544, 320)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Connect RGB preview to the detection network
    colorCam.preview.link(detectionNetwork.input)

# Create and configure the object tracker
objectTracker = pipeline.createObjectTracker()
objectTracker.setDetectionLabelsToTrack([1])  # Track people
# Possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# Take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

# Link detection networks outputs to the object tracker
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)

# Send tracklets to the host
trackerOut = pipeline.createXLinkOut()
trackerOut.setStreamName("tracklets")
objectTracker.out.link(trackerOut.input)

# Send send RGB preview frames to the host
xlinkOut = pipeline.createXLinkOut()
xlinkOut.setStreamName("preview")
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

if SPI:
    # Send tracklets via SPI to the MCU
    spiOut = pipeline.createSPIOut()
    spiOut.setStreamName("tracklets")
    spiOut.setBusId(0)
    objectTracker.out.link(spiOut.input)


class PeopleTracker:
    def __init__(self):
        self.startTime = time.monotonic()
        self.counter = 0
        self.fps = 0
        self.frame = None
        self.color = (255, 0, 0)
        self.tracking = {}
        self.people_counter = [0, 0] # [0] = Y axis (up/down), [1] = X axis (left/right)
        self.statusMap = {
            dai.Tracklet.TrackingStatus.NEW : "NEW",
            dai.Tracklet.TrackingStatus.TRACKED :"TRACKED",
            dai.Tracklet.TrackingStatus.LOST : "LOST",
            dai.Tracklet.TrackingStatus.REMOVED: "REMOVED"}

    def to_planar(self, arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def tracklet_removed(self, coords1, coords2):
        deltaX = coords2[0] - coords1[0]
        deltaY = coords2[1] - coords1[1]

        if abs(deltaX) > abs(deltaY) and abs(deltaX) > THRESH_DIST_DELTA:
            direction = "left" if 0 > deltaX else "right"
            self.people_counter[1] += 1 if 0 > deltaX else -1
            print(f"Person moved {direction}")
            # print("DeltaX: " + str(abs(deltaX)))
        if abs(deltaY) > abs(deltaX) and abs(deltaY) > THRESH_DIST_DELTA:
            direction = "up" if 0 > deltaY else "down"
            self.people_counter[0] += 1 if 0 > deltaY else -1
            print(f"Person moved {direction}")
            # print("DeltaY: " + str(abs(deltaY)))
        # else: print("Invalid movement")

    def get_centroid(self, roi):
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y
        return ((x2-x1)/2+x1, (y2-y1)/2+y1)

    def check_queues(self, preview, tracklets):
        imgFrame = preview.tryGet()
        track = tracklets.tryGet()

        if imgFrame is not None:
            self.counter+=1
            current_time = time.monotonic()
            if (current_time - self.startTime) > 1 :
                self.fps = self.counter / (current_time - self.startTime)
                self.counter = 0
                self.startTime = current_time
            self.frame = imgFrame.getCvFrame()
        if track is not None:
            trackletsData = track.tracklets
            for t in trackletsData:

                # If new tracklet, save its centroid
                if (t.status == dai.Tracklet.TrackingStatus.NEW):
                    self.tracking[str(t.id)] = self.get_centroid(t.roi)
                # If tracklet was removed, check the "path" of this traclet
                if (t.status == dai.Tracklet.TrackingStatus.REMOVED):
                    self.tracklet_removed(self.tracking[str(t.id)], self.get_centroid(t.roi))

                roi = t.roi.denormalize(self.frame.shape[1], self.frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                try:
                    label = labelMap[t.label]
                except:
                    label = t.label

                cv2.putText(self.frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
                cv2.circle(self.frame, (int((x2-x1)/2+ x1), int((y2-y1)/2+ y1)), 4, (0, 255, 0))
                cv2.putText(self.frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
                cv2.putText(self.frame, self.statusMap[t.status], (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), self.color, cv2.FONT_HERSHEY_SIMPLEX)

        if self.frame is not None:
            cv2.putText(self.frame, "NN fps: {:.2f}".format(self.fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, self.color)
            cv2.putText(self.frame, f"Counter X: {self.people_counter[1]}, Counter Y: {self.people_counter[0]}", (3, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,0))
            cv2.imshow("tracker", self.frame)


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    preview = device.getOutputQueue("preview", maxSize=4, blocking=False)
    tracklets = device.getOutputQueue("tracklets", maxSize=4, blocking=False)

    peopleTracker = PeopleTracker()

    if VIDEO:
        videoQ = device.getInputQueue("video_in")

        cap = cv2.VideoCapture(str(Path(videoPath).resolve().absolute()))
        while cap.isOpened():
            read_correctly, video_frame = cap.read()
            if not read_correctly:
                break

            img = dai.ImgFrame()
            # Also reshapes the video frame to 544x320
            img.setData(peopleTracker.to_planar(video_frame, (544, 320)))
            img.setType(dai.RawImgFrame.Type.BGR888p)
            img.setTimestamp(monotonic())
            img.setWidth(544)
            img.setHeight(320)
            videoQ.send(img)

            peopleTracker.check_queues(preview, tracklets)

            if cv2.waitKey(1) == ord('q'):
                break
        print("End of the video")
    else:
        while True:
            peopleTracker.check_queues(preview, tracklets)
            if cv2.waitKey(1) == ord('q'):
                break
