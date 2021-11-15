#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
import blobconverter
import numpy as np
from libraries.depthai_replay import Replay

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay object
replay = Replay(args.path)
# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.init_pipeline()

# Resize color frames prior to sending them to the device
replay.set_resize_color((300, 300))

# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keep_aspect_ratio(True)

nn = pipeline.createMobileNetSpatialDetectionNetwork()
nn.setBoundingBoxScaleFactor(0.3)
nn.setDepthLowerThreshold(100)
nn.setDepthUpperThreshold(5000)

nn.setBlobPath(str(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6)))
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(False)

# Link required inputs to the Spatial detection network
nodes.color.out.link(nn.input)
nodes.stereo.depth.link(nn.inputDepth)

detOut = pipeline.createXLinkOut()
detOut.setStreamName("det_out")
nn.out.link(detOut.input)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

right_s_out = pipeline.createXLinkOut()
right_s_out.setStreamName("rightS")
nodes.stereo.syncedRight.link(right_s_out.input)

left_s_out = pipeline.createXLinkOut()
left_s_out.setStreamName("leftS")
nodes.stereo.syncedLeft.link(left_s_out.input)

with dai.Device(pipeline) as device:
    replay.create_queues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)
    rightS_Q = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)
    leftS_Q = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)

    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.send_frames():
        rgbFrame = replay.lastFrame['color']

        # if mono:
        cv2.imshow("left", leftS_Q.get().getCvFrame())
        cv2.imshow("right", rightS_Q.get().getCvFrame())

        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        inDet = detQ.tryGet()
        if inDet is not None:
            # Display (spatial) object detections on the color frame
            for detection in inDet.detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * 300)
                x2 = int(detection.xmax * 300)
                y1 = int(detection.ymin * 300)
                y2 = int(detection.ymax * 300)
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(rgbFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')

