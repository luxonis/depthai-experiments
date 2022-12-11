#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
from depthai_sdk import FPSHandler
from scipy.special import softmax
from utils import BeamSearchDecoder


# --------------- Arguments ---------------

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn_path', type=str, help="select model blob path for inference", default=None)

args = parser.parse_args()

NN_PATH = args.nn_path

NN_WIDTH = 750
NN_HEIGHT = 256
        
# --------------- Pipeline ---------------
#   MonoCamera => Crop => Resize => NN
# ----------------------------------------

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# MonoCamera
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Crop manip
manip1 = pipeline.create(dai.node.ImageManip)
manip1.initialConfig.setCropRect(0, 0, 0.694444444, 0.355555556)
monoRight.out.link(manip1.inputImage)

# Resize manip
manip2 = pipeline.create(dai.node.ImageManip)
manip2.initialConfig.setResize(750, 256)
manip1.out.link(manip2.inputImage)


# NN classifier
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(NN_PATH)
manip2.out.link(nn.input)

# Send class sequence predictions from the NN to the host via XLink
nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

mono_xout = pipeline.createXLinkOut()
mono_xout.setStreamName("mono")
manip2.out.link(mono_xout.input)


with dai.Device(pipeline) as device:

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,0,255)
    thickness = 1
    
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="mono", maxSize=4, blocking=False)
    
    fps = 0
    fps_handler = FPSHandler()

    characters = "1234567890."
    codec = BeamSearchDecoder(characters, beam_len=30)

    while True:

        fps_handler.tick("nn")
        fps = fps_handler.tickFps("nn")

        out = qNn.get()

        output = np.array(out.getLayerFp16("845")).reshape(24,1,12)

        output = np.transpose(output, [1,0,2])
        classes_softmax = softmax(output, 2)[0]

        pred = codec.decode(classes_softmax)

        frame = qCam.get().getCvFrame()
        cv_img = cv2.putText(frame, f"FPS: {fps:.0f}  " + "Pred " + pred, (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow(
            "7 SEG",
            cv_img
        )
        if cv2.waitKey(1) == ord('q'):
            break