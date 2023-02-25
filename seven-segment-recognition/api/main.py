#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
from scipy.special import softmax

from depthai_sdk.fps import FPSHandler
from utils import BeamSearchDecoder

# --------------- Arguments ---------------

NN_WIDTH = 750
NN_HEIGHT = 256

# --------------- Pipeline ---------------
#   MonoCamera => Crop => Resize => NN
# ----------------------------------------

pipeline = dai.Pipeline()

# MonoCamera
color = pipeline.create(dai.node.ColorCamera)

# Crop manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(750, 256)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)
color.video.link(manip.inputImage)

# NN classifier
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(blobconverter.from_zoo(name="7_segment_recognition_256x750", zoo_type="depthai", shaves=6))
manip.out.link(nn.input)

# Send class sequence predictions from the NN to the host via XLink
nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

mono_xout = pipeline.createXLinkOut()
mono_xout.setStreamName("mono")
manip.out.link(mono_xout.input)

# _xout = pipeline.createXLinkOut()
# _xout.setStreamName("color")
# color.video.link(_xout.input)

with dai.Device(pipeline) as device:
    font = cv2.FONT_HERSHEY_SIMPLEX

    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="mono", maxSize=4, blocking=False)
    # qClr = device.getOutputQueue(name="color")

    fps = FPSHandler()

    characters = "1234567890."
    codec = BeamSearchDecoder(characters, beam_len=30)

    while True:
        out: dai.NNData = qNn.get()
        fps.nextIter()

        output = np.array(out.getLayerFp16("845")).reshape(24, 1, 12)
        output = np.transpose(output, [1, 0, 2])
        classes_softmax = softmax(output, 2)[0]

        # if qClr.has():
        #     frame = qClr.get().getCvFrame()
        #     cv2.imshow('clr', cv2.pyrDown(frame))

        pred = codec.decode(classes_softmax)

        frame = qCam.get().getCvFrame()
        frame = cv2.putText(frame, f"FPS: {fps.fps():.1f}  " + "Pred " + pred, (20, 20), font, 0.5, (0, 0, 0), 3,
                            cv2.LINE_AA)
        frame = cv2.putText(frame, f"FPS: {fps.fps():.1f}  " + "Pred " + pred, (20, 20), font, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        cv2.imshow("7 Segment display recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break
