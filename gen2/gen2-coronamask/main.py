#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. Use either \"-cam\" to run on RGB camera as a source or \"-vid <path>\" to run on video"
    )

debug = not args.no_debug

def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(False)
    camRgb.setFps(40)

    # Define a neural network that will make predictions based on the source frames
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(str(Path("models/model.blob").resolve().absolute()))
    nn.setNumInferenceThreads(2)
    # nn.input.setBlocking(False)

    if args.camera:
        camRgb.preview.link(nn.input)
    else:
        detection_in = pipeline.create(dai.node.XLinkIn)
        detection_in.setStreamName("detection_in")
        detection_in.out.link(nn.input)

    # Create outputs
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    nn.out.link(nnOut.input)

    return pipeline


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

labelMap = ["background", "no mask", "mask", "no mask"]
if args.camera:
    fps = FPSHandler()
else:
    cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
    fps = FPSHandler(cap)

# Pipeline defined, now the device is connected to
with dai.Device(create_pipeline()) as device:
    if args.camera:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    else:
        detIn = device.getInputQueue("detection_in")

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    if args.video:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))

    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def shouldRun():
        return cap.isOpened() if args.video else True

    def getFrame():
        if args.video:
            return cap.read()
        else:
            return True, qRgb.get().getCvFrame()

    while shouldRun():
        read_correctly, frame = getFrame()

        if not read_correctly:
            break

        fps.next_iter()

        if not args.camera:
            tstamp = time.monotonic()
            lic_frame = dai.ImgFrame()
            lic_frame.setData(to_planar(frame, (300, 300)))
            lic_frame.setTimestamp(tstamp)
            lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
            lic_frame.setWidth(300)
            lic_frame.setHeight(300)
            detIn.send(lic_frame)

        detections = qDet.get().detections
        cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
