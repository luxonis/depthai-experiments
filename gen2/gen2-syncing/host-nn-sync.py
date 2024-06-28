#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np

class HostSeqSync:
    def __init__(self):
        self.msgs = []
    def add_frame(self, msg):
        self.msgs.append(msg)
    def get_frame(self, seq):
        for i, frame in enumerate(self.msgs):
            if seq == frame.getSequenceNum():
                self.msgs = self.msgs[i:] # Remove previous frames
                break
        return self.msgs[0]

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setVideoSize(900, 900) # High res frames for visualization on host
camRgb.setPreviewSize(300,300)
camRgb.setInterleaved(False)
camRgb.setFps(40)

# Send frames to the host
xout_frame = pipeline.create(dai.node.XLinkOut)
xout_frame.setStreamName("frame")
camRgb.video.link(xout_frame.input)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
# Frames will come a lot faster than NN will be able to process. To always
# run inference on latest frame, we can set its queue size to 1.
# Default is Blocking and queue size 5.
detection_nn.input.setBlocking(False)
detection_nn.input.setQueueSize(1)
camRgb.preview.link(detection_nn.input) # Downscaled 300x300 frames

# Send bounding boxes to the host
xout_land = pipeline.create(dai.node.XLinkOut)
xout_land.setStreamName("det")
detection_nn.out.link(xout_land.input)

# Upload the pipeline to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    sync = HostSeqSync()

    while True:
        if q_frame.has():
            sync.add_frame(q_frame.get())

        if q_det.has():
            detIn = q_det.get()
            frame = sync.get_frame(detIn.getSequenceNum()).getCvFrame()
            dets = detIn.detections

            for detection in dets:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
