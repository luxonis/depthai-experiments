#!/usr/bin/env python3
import argparse
import blobconverter
import depthai as dai

from object_localizer import ObjectLocalizer

'''
Mobile object localizer demo running on device on RGB camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Link to the original model:
https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

Blob taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float, help="Confidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH = blobconverter.from_zoo(name="mobile_object_localizer_192x192", zoo_type="depthai", shaves=6)
NN_WIDTH = 192
NN_HEIGHT = 192
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

print("Creating pipeline...")
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(40)

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(NN_PATH)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    manip.initialConfig.setKeepAspectRatio(False)

    cam.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

    object_localizer = pipeline.create(ObjectLocalizer).build(cam.preview, detection_nn.out, manip.out)
    object_localizer.set_threshold(THRESHOLD)

    print("Running pipeline...")
    pipeline.run()
    print("Pipeline exited.")