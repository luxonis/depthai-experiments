#!/usr/bin/env python3
import argparse
import depthai as dai

from object_localizer import ObjectLocalizer


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Confidence threshold", default=0.2)
args = parser.parse_args()

THRESHOLD = args.threshold
NN_WIDTH = 512
NN_HEIGHT = 288
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

device = dai.Device()
platform = device.getPlatform()
model_description = dai.NNModelDescription(modelSlug="yolov6-nano", platform=platform.name, modelVersionSlug="r2-coco-512x288")
archive_path = dai.getModelFromZoo(model_description)

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    preview = cam.requestOutput((PREVIEW_WIDTH, PREVIEW_HEIGHT), dai.ImgFrame.Type.RGB888i)
    nn_input = cam.requestOutput((NN_WIDTH, NN_HEIGHT), dai.ImgFrame.Type.BGR888i if platform == dai.Platform.RVC4 else dai.ImgFrame.Type.BGR888p)

    nnarchive = dai.NNArchive(archive_path)

    detection_nn = pipeline.create(dai.node.DetectionNetwork).build(nn_input, nnarchive)

    object_localizer = pipeline.create(ObjectLocalizer).build(
        cam=preview,
        nn=detection_nn.out, 
        manip=nn_input
    )
    object_localizer.set_threshold(THRESHOLD)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline exited.")