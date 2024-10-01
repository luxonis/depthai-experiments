#!/usr/bin/env python3
import argparse

import depthai as dai
from host_node.blur_bboxes import BlurBboxes
from host_node.host_display import Display
from host_node.host_fps_drawer import FPSDrawer
from host_node.normalize_bbox import NormalizeBbox
from mask_to_bbox import MaskToBbox
from text_detection_parser import TextDetectionParser

device = dai.Device()
platform = device.getPlatform()
rvc2 = platform == dai.Platform.RVC2

NN_WIDTH, NN_HEIGHT = (256, 256) if rvc2 else (960, 544)

modelDescription = dai.NNModelDescription(modelSlug="paddle-text-detection", platform=device.getPlatform().name, modelVersionSlug=f"{NN_HEIGHT}x{NN_WIDTH}")
archivePath = dai.getModelFromZoo(modelDescription)
nn_archive = dai.NNArchive(archivePath)

parser = argparse.ArgumentParser()
parser.add_argument("-bt", "--box_thresh", help="set the confidence threshold of boxes", default=0.2, type=float)
parser.add_argument("-t", "--thresh", help="set the bitmap threshold", default=0.01, type=float)
parser.add_argument("-ms", "--min_size", default=1, type=int, help='set min size of box')
parser.add_argument("-mc", "--max_candidates", default=75, type=int, help='maximum number of candidate boxes')


args = parser.parse_args()

with dai.Pipeline(device) as pipeline:
    # Define camera
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = cam.requestOutput((NN_WIDTH, NN_HEIGHT), dai.ImgFrame.Type.BGR888p if rvc2 else dai.ImgFrame.Type.BGR888i)

    # Define a neural network that will detect text
    detection_nn = pipeline.create(dai.node.NeuralNetwork).build(output, nn_archive)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)
    parser = pipeline.create(TextDetectionParser).build(nn=detection_nn.out, nn_size=(NN_HEIGHT, NN_WIDTH))
    parser.setConfidenceThreshold(args.thresh)

    mask_to_bbox = pipeline.create(MaskToBbox).build(nn=parser.output)
    mask_to_bbox.setBboxThreshold(args.box_thresh)
    mask_to_bbox.setBitmapThreshold(args.thresh)
    mask_to_bbox.setMinSize(args.min_size)
    mask_to_bbox.setMaxCandidates(args.max_candidates)
    mask_to_bbox.setPadding(5)
    normalize_bbox = pipeline.create(NormalizeBbox).build(nn=mask_to_bbox.output, frame=output, manip_mode=dai.ImgResizeMode.CROP)
    blur_bboxes = pipeline.create(BlurBboxes).build(nn=normalize_bbox.output, frame=output)

    output_fps = pipeline.create(FPSDrawer).build(blur_bboxes.output)
    display_preds = pipeline.create(Display).build(output_fps.output)
    display_preds.setName("Text Blur")

    print("Pipeline created")
    pipeline.run()
    print("Pipeline ended")
