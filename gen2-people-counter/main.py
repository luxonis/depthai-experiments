#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
from time import monotonic
import os

parentDir = Path(__file__).parent

#=====================================================================================
# To use a different NN, change `size` and `nnPath` here:
size = (544, 320)
nnPath = parentDir / Path(f"models/person-detection-retail-0013_2021.3_7shaves.blob")
#=====================================================================================

# Labels
labelMap = ["background", "person"]

# Get argument first
parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help=".blob path")
parser.add_argument('-i', '--image', type=str,
                    help="Path to an image file to be used for inference (conflicts with -cam)")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI RGB camera for inference (conflicts with -vid)")
args = parser.parse_args()

# Whether we want to use images from host or rgb camera
IMAGE = not args.camera

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Crop the frame to desired aspect ratio
# size (width, heigth)
def crop_frame(frame, size):
    shape = frame.shape
    h = shape[0]
    w = shape[1]
    current_ratio = w / h
    new_ratio = size[0] / size[1]

    # Crop width/heigth to match the aspect ratio needed by the NN
    if new_ratio < current_ratio:  # Crop width
        # Use full height, crop width
        new_w = (new_ratio/current_ratio) * w
        crop = int((w - new_w) / 2)
        preview = frame[:, crop:w-crop]
    else:  # Crop height
        # Use full width, crop height
        new_h = (current_ratio/new_ratio) * h
        crop = int((h - new_h) / 2)
        preview = frame[crop:h-crop, :]
    # To planar
    print("new shape", preview.shape)
    return preview


def to_planar(frame, shape):
    return cv2.resize(frame, shape).transpose(2, 0, 1).flatten()


# Start defining a pipeline
pipeline = dai.Pipeline()

# Create and configure the detection network
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
detectionNetwork.setBlobPath(str(Path(nnPath).resolve().absolute()))
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)

if IMAGE:
    # Configure XLinkIn - we will send img frames through it
    imgIn = pipeline.createXLinkIn()
    imgIn.setStreamName("img_in")
    imgIn.out.link(detectionNetwork.input)

else:
    # Create and configure the color camera
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(size[0], size[1])
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Connect RGB preview to the detection network
    colorCam.preview.link(detectionNetwork.input)

    # Send send RGB preview frames to the host
    rgbOut = pipeline.createXLinkOut()
    rgbOut.setStreamName("preview")
    detectionNetwork.passthrough.link(rgbOut.input)

detOut = pipeline.createXLinkOut()
detOut.setStreamName("detections")
detectionNetwork.out.link(detOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start the pipeline
    device.startPipeline()
    detectionQ = device.getOutputQueue("detections", maxSize=4, blocking=False)

    if IMAGE:
        imgQ = device.getInputQueue("img_in")
        if args.image:
            imgPaths = [args.image]
        else:
            imgNames = list(os.listdir('./images'))
            imgPaths = [parentDir / Path('images') / name for name in imgNames]
        og_frames = [crop_frame(cv2.imread(str(imgPath)), size) for imgPath in imgPaths]
    else:
        rgbQ = device.getOutputQueue("preview", maxSize=4, blocking=False)

    i = 0
    while True:
        if IMAGE:
            print("new index", i)
            og_frame = og_frames[i]
            i += 1
            if len(og_frames)-1 < i: i = 0

            img = dai.ImgFrame()
            frame = og_frame.copy()
            img.setData(to_planar(frame, size))
            img.setType(dai.RawImgFrame.Type.BGR888p)
            img.setTimestamp(monotonic())
            img.setWidth(size[0])
            img.setHeight(size[1])
            imgQ.send(img)
        else:
            frame = rgbQ.get().getCvFrame()

        detIn = detectionQ.get()
        for detection in detIn.detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.putText(frame, f"People count: {len(detIn.detections)}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))

        cv2.imshow("color", frame)

        if cv2.waitKey(3000 if IMAGE else 1) == ord('q'):
            break
