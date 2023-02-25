#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(900, 900) # High res frames for visualization on host
camRgb.setInterleaved(False)
camRgb.setFps(40)

downscale_manip = pipeline.create(dai.node.ImageManip)
downscale_manip.initialConfig.setResize(300, 300)
camRgb.preview.link(downscale_manip.inputImage)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
# Frames will come a lot faster than NN will be able to process. To always
# run inference on latest frame, we can set its queue size to 1.
# Default is Blocking and queue size 5.
detection_nn.input.setBlocking(False)
detection_nn.input.setQueueSize(1)
downscale_manip.out.link(detection_nn.input) # Downscaled 300x300 frames

# Script node will sync high-res frames with ImgDetections
# Object detections will get synced with the frames.
script = pipeline.create(dai.node.Script)
# Stream NN detections to Script npode
detection_nn.out.link(script.inputs["nn"])
# Stream high-res frames to Script node
camRgb.preview.link(script.inputs["frames"])
# script.inputs["frames"].setBlocking(False)
# Currenly there are no bindings for ImgDetections.getSequenceNum() inside Script node
detection_nn.passthrough.link(script.inputs['passthrough'])

script.setScript("""
l = [] # List of assets

# So the correct frame will be the first in the list
def get_latest_frame(seq):
    global l
    for i, frame in enumerate(l):
        if seq == frame.getSequenceNum():
            # node.warn(f"List len {len(l)} Frame with same seq num: {i},seq {seq}")
            l = l[i:]
            break
    return l[0]

while True:
    l.append(node.io['frames'].get())

    face_dets = node.io['nn'].tryGet()
    # node.warn(f"Faces detected: {len(face_dets)}")
    if face_dets is not None:
        seq = node.io['passthrough'].get().getSequenceNum()
        if len(l) == 0: continue

        # Sync detection with frame and send them both to the host
        node.io['img_out'].send(get_latest_frame(seq))
        node.io['det_out'].send(face_dets)

        l.pop(0) # Remove matching frame from the list
""")

# Send frames to the host
xout_frame = pipeline.create(dai.node.XLinkOut)
xout_frame.setStreamName("frame")
script.outputs['img_out'].link(xout_frame.input)

# Send bounding boxes to the host
xout_land = pipeline.create(dai.node.XLinkOut)
xout_land.setStreamName("det")
script.outputs['det_out'].link(xout_land.input)

# Upload the pipeline to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_frame = device.getOutputQueue(name="frame", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det", maxSize=4, blocking=False)

    while True:
        frame = q_frame.get().getCvFrame()
        dets = q_det.get().detections

        for detection in dets:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
