#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import errno
import os
from sklearn.cluster import DBSCAN


'''
LaneNet lane detection demo running on device with video input from host.
Post-processing is not the same as in official paper!
Please refer to https://arxiv.org/abs/1802.05591 for how postprocessing should be done!
Run as:
python3 -m pip install -r requirements.txt
python3 download.py
python3 main.py -v path/to/video

Model is taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/141_lanenet-lane-detection
and converted to blob with required flags.

DepthAI 2.9.0.0 is required. Blob was compiled using OpenVino 2021.4
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/lanenet_openvino_2021.4_6shave.blob', type=str)
parser.add_argument('-v', '--video_path', help="Path to video frame", default="vids/vid3.mp4")

args = parser.parse_args()

video_source = args.video_path
nn_path = args.nn_model

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 512, 256

# set max clusters for color output (number of lines + a few more due to errors in fast postprocessing)
MAX_CLUSTERS = 6


# --------------- Methods ---------------

# perform postprocessing - clustering of line embeddings using DBSCAN
# note that this is not whole postprocessing, just a quick implementation to show what is possible
# if you want to use whole postprocessing please refer to the LaneNet paper: https://arxiv.org/abs/1802.05591
def cluster_outputs(binary_seg_ret, instance_seg_ret):
    # create mask from binary output
    mask = binary_seg_ret.copy()
    mask = mask.astype(bool)

    # mask out embeddings
    embeddings = instance_seg_ret.copy()
    embeddings = np.transpose(embeddings, (1,0))
    embeddings_masked = embeddings[mask]

    # sort so same classes are sorted first each time and generate inverse sort
    # works only if new lanes are added on the right side
    idx = np.lexsort(np.transpose(embeddings_masked)[::-1])
    idx_inverse = np.empty_like(idx)
    idx_inverse[idx] = np.arange(idx.size)
    embeddings_masked = embeddings_masked[idx]

    # cluster embeddings with DBSCAN
    clustering = DBSCAN(eps=0.4, min_samples=500, algorithm="kd_tree").fit(embeddings_masked)

    # unsort so pixels match their positions again
    clustering_labels = clustering.labels_[idx_inverse]

    # create an array of masked clusters
    clusters = np.zeros((NN_WIDTH * NN_HEIGHT,))
    clusters[mask] = clustering_labels + 1
    clusters = clusters.reshape((NN_HEIGHT, NN_WIDTH))

    return clusters

# create overlay from cluster_outputs
def create_overlay(cluster_outputs):
    output = np.array(cluster_outputs) * (255 / MAX_CLUSTERS) # multiply to get classes between 0 and 255
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0, 0, 0]
    return output_colors

# merge 2 frames together
def show_output(overlay, frame):
    return cv2.addWeighted(frame, 1, overlay, 0.8, 0)


# --------------- Check input ---------------
vid_path = Path(video_source)
if not vid_path.is_file():
    raise FileNotFoundError("Video file not found. Either run download.py script first, or specify the path to the video file with '--video_path' argument.")

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Create Manip for image resizing and NN for count inference
manip = pipeline.create(dai.node.ImageManip)
detection_nn = pipeline.create(dai.node.NeuralNetwork)

# Create output links, and in link for video
manipOut = pipeline.create(dai.node.XLinkOut)
xinFrame = pipeline.create(dai.node.XLinkIn)
xlinkOut = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

manipOut.setStreamName("manip")
xinFrame.setStreamName("inFrame")
xlinkOut.setStreamName("trackerFrame")
nnOut.setStreamName("nn")

# Properties
manip.initialConfig.setResizeThumbnail(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.inputImage.setBlocking(True)

# setting node configs
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Linking
manip.out.link(manipOut.input)
manip.out.link(detection_nn.input)
xinFrame.out.link(manip.inputImage)
detection_nn.out.link(nnOut.input)

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    qIn = device.getInputQueue(name="inFrame", maxSize=1, blocking=False)
    qManip = device.getOutputQueue(name="manip", maxSize=4)
    qNN = device.getOutputQueue(name="nn", maxSize=4)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    detections = []
    frame = None


    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    cap = cv2.VideoCapture(args.video_path)
    baseTs = time.monotonic()
    simulatedFps = 30
    inputFrameShape = (NN_WIDTH, NN_HEIGHT)

    while cap.isOpened():
        # read, process image and send it to NN
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888p)
        img.setData(to_planar(frame, inputFrameShape))
        img.setTimestamp(baseTs)
        baseTs += 1 / simulatedFps

        img.setWidth(inputFrameShape[0])
        img.setHeight(inputFrameShape[1])
        qIn.send(img)

        # get resized image and NN output queues
        manip = qManip.get()
        inNN = qNN.get()

        # get manip frame
        manipFrame = manip.getCvFrame()

        # read output
        # first layer is a binary segmentation mask of lanes
        out1 = np.array(inNN.getLayerInt32("LaneNet/bisenetv2_backend/binary_seg/ArgMax/Squeeze"))
        out1 = out1.astype(np.uint8)
        # second layer is an array of embeddings of dimension 4 for each pixel in the input
        out2 = np.array(inNN.getLayerFp16("LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/Conv2D"))
        out2 = out2.reshape((4, NN_WIDTH * NN_HEIGHT))

        # cluster outputs
        clusters = cluster_outputs(out1, out2)

        overlay = create_overlay(clusters)

        # show fps
        color_black, color_white = (0,0,0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(manipFrame, (0,manipFrame.shape[0]-h1-6), (w1 + 2, manipFrame.shape[0]), color_white, -1)
        cv2.putText(manipFrame, label_fps, (2, manipFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)


        # add overlay
        outFrame = show_output(overlay, manipFrame)
        cv2.imshow("Lane prediction", outFrame)

        # FPS counter
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        if cv2.waitKey(1) == ord('q'):
            break
