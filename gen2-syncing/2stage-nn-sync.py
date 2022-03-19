#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

class TwoStageHostSeqSync:
    def __init__(self):
        self.frames = []
    def add_msg(self, msg):
        self.frames.append(msg)
    def get_msg(self, target_seq):
        for i, imgFrame in enumerate(self.frames):
            if target_seq == imgFrame.getSequenceNum():
                self.frames = self.frames[i:]
                break
        return self.frames[0]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(900, 900) # High res frames for visualization on host
camRgb.setInterleaved(False)

# Send frames to the host
xout_frame = pipeline.create(dai.node.XLinkOut)
xout_frame.setStreamName("det_frame")
camRgb.preview.link(xout_frame.input)

downscale_manip = pipeline.create(dai.node.ImageManip)
downscale_manip.initialConfig.setResize(300, 300)
camRgb.preview.link(downscale_manip.inputImage)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
downscale_manip.out.link(detection_nn.input) # Downscaled 300x300 frames

# Send detections to the host
xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("det_nn")
detection_nn.out.link(xout_det.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'landmarks_manip' to crop the initial frame.
# Object detections will get synced with the frames.
script = pipeline.create(dai.node.Script)
# Stream NN detections to Script npode
detection_nn.out.link(script.inputs["nn"])
# Stream high-res frames to Script node
camRgb.preview.link(script.inputs["frames"])
# Currenly there are no bindings for ImgDetections.getSequenceNum() inside Script node
detection_nn.passthrough.link(script.inputs['passthrough'])

script.setScript("""
l = [] # List of images

# So the correct frame will be the first in the list
def get_latest_frame(seq):
    global l
    for i, frame in enumerate(l):
        if seq == frame.getSequenceNum():
            # node.warn(f"List len {len(l)} Frame with same seq num: {i},seq {seq}")
            l = l[i:]
            break
    return l[0]

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999

while True:
    l.append(node.io['frames'].get())

    face_dets = node.io['nn'].tryGet()
    # node.warn(f"Faces detected: {len(face_dets)}")
    if face_dets is not None:
        seq = node.io['passthrough'].get().getSequenceNum()

        if len(l) == 0: continue
        img = get_latest_frame(seq)

        for det in face_dets.detections:
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(48, 48)
            cfg.setKeepAspectRatio(False)
            node.io['manip_img'].send(img)
            node.io['manip_cfg'].send(cfg)
""")

landmarks_manip = pipeline.create(dai.node.ImageManip)
landmarks_manip.initialConfig.setResize(48, 48)
landmarks_manip.setWaitForConfigInput(False)
script.outputs['manip_cfg'].link(landmarks_manip.inputConfig)
script.outputs['manip_img'].link(landmarks_manip.inputImage)

landmarks_nn = pipeline.create(dai.node.NeuralNetwork)
landmarks_nn.setBlobPath(blobconverter.from_zoo(name="landmarks-regression-retail-0009", shaves=6))
landmarks_manip.out.link(landmarks_nn.input)

# Send landmark NN results to the host
xout_land = pipeline.create(dai.node.XLinkOut)
xout_land.setStreamName("land_nn")
landmarks_nn.out.link(xout_land.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_frame = device.getOutputQueue(name="det_frame", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det_nn", maxSize=4, blocking=False)
    q_land = device.getOutputQueue(name="land_nn", maxSize=4, blocking=False)

    results = []
    sync = HostSeqSync()

    def calc_coords(bbox, landmarks):
        x = int((bbox[2] - bbox[0]) * landmarks[0] + bbox[0])
        y = int((bbox[3] - bbox[1]) * landmarks[1] + bbox[1])
        return (x,y)

    while True:
        if q_frame.has(): sync.add_msg(q_frame.get())

        if q_det.has():
            inDet = q_det.get()
            frame = sync.get_msg(inDet.getSequenceNum()).getCvFrame()

            detections = inDet.detections
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # If there is a face detected, there will also be landmarks
                # inference result available soon, so we can wait for it
                inlandmarks = q_land.getAll()[0]
                if inlandmarks.getSequenceNum() != inDet.getSequenceNum():
                    print(f"{inlandmarks.getSequenceNum()}, det: {inDet.getSequenceNum()}")

                landmarks = inlandmarks.getFirstLayerFp16()

                while not len(results) < len(detections) and len(results) > 0:
                    results.pop(0)
                results.append({
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "ts": time.time()
                })


            results = list(filter(lambda result: time.time() - result["ts"] < 0.2, results))

            for result in results:
                bbox = result["bbox"]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                y = (bbox[1] + bbox[3]) // 2
                cv2.circle(frame, calc_coords(bbox, result['landmarks'][:2]), 3, (255, 255, 0),3)  # Right eye
                cv2.circle(frame, calc_coords(bbox, result['landmarks'][2:4]), 3, (0, 255, 255),3)  # Left eye
                cv2.circle(frame, calc_coords(bbox, result['landmarks'][4:6]), 3, (255, 0, 255),3)  # Nose
                cv2.circle(frame, calc_coords(bbox, result['landmarks'][6:8]), 3, (255, 0, 0),3)  # Right mouth
                cv2.circle(frame, calc_coords(bbox, result['landmarks'][8:]), 3, (0, 255, 0),3)  # Left mouth

                # You could also get result["3d"].x and result["3d"].y coordinates
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
