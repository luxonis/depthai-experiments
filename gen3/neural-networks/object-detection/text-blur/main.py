#!/usr/bin/env python3
import depthai as dai
import argparse
from display_blur_text import DisplayBlurText
from host_display import Display
from host_fps_drawer import FPSDrawer

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

    blur_text = pipeline.create(DisplayBlurText).build(
        camOut=output,
        nnOut=detection_nn.out,
        nn_size=(NN_HEIGHT, NN_WIDTH)
    )
    blur_text.setBboxThreshold(args.box_thresh)
    blur_text.setBitmapThreshold(args.thresh)
    blur_text.setMinSize(args.min_size)
    blur_text.setMaxCandidates(args.max_candidates)

    display_preds = pipeline.create(Display).build(blur_text.output_preds)
    display_preds.setName("Text Predictions")

    output_fps = pipeline.create(FPSDrawer).build(blur_text.output)

    display = pipeline.create(Display).build(output_fps.output)
    display.setName("Text Blur")

    print("Pipeline created")
    pipeline.run()
    print("Pipeline ended")
