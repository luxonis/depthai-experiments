#!/usr/bin/env python3
import time
import argparse
import blobconverter
import depthai as dai
from face_detection import FaceDetection

model_description = dai.NNModelDescription(modelSlug="yunet", platform="RVC2", modelVersionSlug="640x640")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.6, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.3, type=float)
parser.add_argument("-topk", "--keep_top_k", default=750, type=int, help='set keep_top_k for results outputing.')
args = parser.parse_args()

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 640, 640
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480

# --------------- Pipeline ---------------
# Start defining a pipeline
with dai.Pipeline() as pipeline:

    # Define a neural network that will detect faces
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setNNArchive(nn_archive)
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
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.inputConfig.setWaitForMessage(False)
    manip.setMaxOutputFrameSize(NN_HEIGHT * NN_WIDTH * 3)

    cam.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

    #Define host node
    face_detection = pipeline.create(FaceDetection).build(
        preview=cam.preview, 
        detection_network=detection_nn.out,
        nn_width=NN_WIDTH,
        nn_height=NN_HEIGHT,
        video_width=VIDEO_WIDTH,
        video_height=VIDEO_HEIGHT)
    face_detection.set_start_time(time.time())
    face_detection.set_confidence_thresh(args.confidence_thresh)
    face_detection.set_iou_thresh(args.iou_thresh)
    face_detection.set_keep_top_k(args.keep_top_k)
    
    pipeline.run()