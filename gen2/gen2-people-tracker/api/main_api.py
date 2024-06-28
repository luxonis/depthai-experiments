#!/usr/bin/env python3

from pathlib import Path
import json

import blobconverter
import cv2
import depthai as dai
import numpy as np
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

videoPath = args.video or parentDir / Path('example_01.mp4')
nnPath = args.nn or blobconverter.from_zoo(name="person-detection-retail-0013", shaves=7)

# Whether we want to use video from host or rgb camera
VIDEO = not args.camera

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 6, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 2, self.line_type)

# Start defining a pipeline
pipeline = dai.Pipeline()

# Create and configure the detection network
detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
detectionNetwork.setBlobPath(str(Path(nnPath).resolve().absolute()))
detectionNetwork.setConfidenceThreshold(0.5)

if VIDEO:
    # Configure XLinkIn - we will send video through it
    videoIn = pipeline.create(dai.node.XLinkIn)
    videoIn.setStreamName("video_in")

    # NOTE: video must be of size 544x320. We will resize this on the
    # host, but you could also use ImageManip node to do it on device

    # Link video in with the detection network
    videoIn.out.link(detectionNetwork.input)

else:
    # Create and configure the color camera
    colorCam = pipeline.create(dai.node.ColorCamera)
    colorCam.setPreviewSize(544, 320)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Connect RGB preview to the detection network
    colorCam.preview.link(detectionNetwork.input)

# Create and configure the object tracker
objectTracker = pipeline.create(dai.node.ObjectTracker)
objectTracker.setDetectionLabelsToTrack([1])  # Track people
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# Take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Link detection networks outputs to the object tracker
detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)


script = pipeline.create(dai.node.Script)
objectTracker.out.link(script.inputs['tracklets'])

with open("script.py", "r") as f:
    s = f.read()
    s = s.replace("THRESH_DIST_DELTA", str(args.threshold))
    script.setScript(s)

# Send tracklets to the host
trackerOut = pipeline.create(dai.node.XLinkOut)
trackerOut.setStreamName("out")
script.outputs['out'].link(trackerOut.input)

# Send send RGB preview frames to the host
xlinkOut = pipeline.create(dai.node.XLinkOut)
xlinkOut.setStreamName("preview")
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

if args.spi:
    # Send tracklets via SPI to the MCU
    spiOut = pipeline.create(dai.node.SPIOut)
    spiOut.setStreamName("tracklets")
    spiOut.setBusId(0)
    objectTracker.out.link(spiOut.input)


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    previewQ = device.getOutputQueue("preview")
    outQ = device.getOutputQueue("out")

    counters = None
    frame = None
    text = TextHelper()

    def update():
        global counters, frame, text
        if previewQ.has():
            frame = previewQ.get().getCvFrame()

        if outQ.has():
            jsonText = str(outQ.get().getData(), 'utf-8')
            counters = json.loads(jsonText)
            print(counters)

        if counters is not None:
            text.putText(frame, f"Up: {counters['up']}, Down: {counters['down']}", (3, 30))
        if frame is not None:
            cv2.imshow("frame", frame)

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    if VIDEO:
        videoQ = device.getInputQueue("video_in")

        cap = cv2.VideoCapture(str(Path(videoPath).resolve().absolute()))
        while cap.isOpened():
            read_correctly, video_frame = cap.read()
            if not read_correctly:
                break

            img = dai.ImgFrame()
            # Also reshapes the video frame to 544x320
            img.setData(to_planar(video_frame, (544, 320)))
            img.setType(dai.RawImgFrame.Type.BGR888p)
            img.setTimestamp(monotonic())
            img.setWidth(544)
            img.setHeight(320)
            videoQ.send(img)

            update()
            if cv2.waitKey(1) == ord('q'):
                break
        print("End of the video")
    else:
        while True:
            update()
            if cv2.waitKey(1) == ord('q'):
                break
