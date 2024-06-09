#!/usr/bin/env python3
"""
The code is edited from gen2-human-pose (depthai-experiments).
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import requests
from time import monotonic
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
import threading

W, H = (456, 256)
running = True
pose = None
keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

def show(frame):
    global keypoints_list, detected_keypoints, personwiseKeypoints

    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        scale_factor = frame.shape[0] / H
        offset_w = int(frame.shape[1] - W * scale_factor) // 2

        def scale(point):
            return int(point[0] * scale_factor) + offset_w, int(point[1] * scale_factor)

        for i in range(18):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame, scale(detected_keypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)


def decode_thread(in_queue):
    global keypoints_list, detected_keypoints, personwiseKeypoints

    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (W, H))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w=W, h=H, detected_keypoints=new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='human-pose-estimation-0001', type=str)
parser.add_argument("-v", "--video", help="Path to video file", default='keypoint_detection/pexels_walk_1.mp4', type=str)
args = parser.parse_args()

nnpath = str(blobconverter.from_zoo(args.model, shaves = 8, use_cache=True))

pipeline = dai.Pipeline()
inputVideo = pipeline.create(dai.node.XLinkIn)
keypoint_network = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

inputVideo.setStreamName("inFrame")
nnOut.setStreamName("nn")

# Network specific settings
keypoint_network.setBlobPath(nnpath)
keypoint_network.setNumInferenceThreads(2)
keypoint_network.input.setBlocking(False)

# Linking
inputVideo.out.link(keypoint_network.input)
keypoint_network.out.link(nnOut.input)

# with open('pipeline.json', 'w') as outfile:
#     json.dump(pipeline.serializeToJson(), outfile, indent=4)

with dai.Device(pipeline) as device:
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
    
    # Input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    t = threading.Thread(target=decode_thread, args=(qDet,))
    t.start()
    
    cap = cv2.VideoCapture(args.video)
    frame_counter = 0
    while cap.isOpened():
        start = time.time()
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #     frame_counter = 0 #Or whatever as long as it is the same as next line
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (W, H)))
        img.setTimestamp(monotonic())
        img.setWidth(W)
        img.setHeight(H)
        qIn.send(img)

        inDet = qDet.tryGet()
        if inDet is not None:
            print(inDet)
            
        show(frame)

t.join()
