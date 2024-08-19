#!/usr/bin/env python3
import argparse
import depthai as dai

from object_localizer import ObjectLocalizer

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Confidence threshold", default=0.2)
args = parser.parse_args()

THRESHOLD = args.threshold
NN_WIDTH = 300
NN_HEIGHT = 300
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(40)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    manip.initialConfig.setKeepAspectRatio(True)

    nnarchive = dai.NNArchive(archive_path)

    detection_nn = pipeline.create(dai.node.DetectionNetwork)
    detection_nn.setNNArchive(nnarchive)
    
    detection_nn.setConfidenceThreshold(THRESHOLD)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    cam.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

    object_localizer = pipeline.create(ObjectLocalizer).build(
        cam=cam.preview, 
        nn=detection_nn.out, 
        manip=manip.out
    )
    object_localizer.set_threshold(THRESHOLD)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline exited.")