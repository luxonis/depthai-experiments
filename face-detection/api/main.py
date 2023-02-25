#!/usr/bin/env python3

import argparse
import time

import blobconverter
import cv2
import depthai as dai
import numpy as np

from utils.priorbox import PriorBox
from utils.utils import draw

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.6, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.3, type=float)
parser.add_argument("-topk", "--keep_top_k", default=750, type=int, help='set keep_top_k for results outputing.')
args = parser.parse_args()

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 160, 120
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a neural network that will detect faces
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(blobconverter.from_zoo(name="face_detection_yunet_160x120", zoo_type="depthai", shaves=6))
detection_nn.input.setBlocking(False)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
cam.setInterleaved(False)
cam.setFps(60)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip.inputConfig.setWaitForMessage(False)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

cam.preview.link(manip.inputImage)
cam.preview.link(xout_cam.input)
manip.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()

        # get all layers
        conf = np.array(in_nn.getLayerFp16("conf")).reshape((1076, 2))
        iou = np.array(in_nn.getLayerFp16("iou")).reshape((1076, 1))
        loc = np.array(in_nn.getLayerFp16("loc")).reshape((1076, 14))

        # decode
        pb = PriorBox(input_shape=(NN_WIDTH, NN_HEIGHT), output_shape=(frame.shape[1], frame.shape[0]))
        dets = pb.decode(loc, conf, iou, args.confidence_thresh)

        # NMS
        if dets.shape[0] > 0:
            # NMS from OpenCV
            bboxes = dets[:, 0:4]
            scores = dets[:, -1]

            keep_idx = cv2.dnn.NMSBoxes(
                bboxes=bboxes.tolist(),
                scores=scores.tolist(),
                score_threshold=args.confidence_thresh,
                nms_threshold=args.iou_thresh,
                eta=1,
                top_k=args.keep_top_k)  # returns [box_num, class_num]

            keep_idx = np.squeeze(keep_idx)  # [box_num, class_num] -> [box_num]
            dets = dets[keep_idx]

        # Draw
        if dets.shape[0] > 0:

            if dets.ndim == 1:
                dets = np.expand_dims(dets, 0)

            img_res = draw(
                img=frame,
                bboxes=dets[:, :4],
                landmarks=np.reshape(dets[:, 4:14], (-1, 5, 2)),
                scores=dets[:, -1]
            )

        # show fps
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

        # show frame
        cv2.imshow("Detections", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
