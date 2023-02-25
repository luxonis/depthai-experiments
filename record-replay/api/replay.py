#!/usr/bin/env python3
import argparse

import blobconverter
import cv2
import depthai as dai
import numpy as np

from depthai_sdk import Replay
from depthai_sdk.utils import frameNorm, cropToAspectRatio

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay objects
replay = Replay(args.path)

replay.disableStream('depth')  # In case depth was saved (mcap)
# Resize color frames prior to sending them to the device
replay.setResizeColor((304, 304))
# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keepAspectRatio(True)

# Initializes the pipeline. This will create required XLinkIn's and connect them together
# Creates StereoDepth node, if both left and right streams are recorded
pipeline, nodes = replay.initPipeline()

nodes.stereo.setSubpixel(True)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(300, 300)
manip.setMaxOutputFrameSize(300 * 300 * 3)
nodes.color.out.link(manip.inputImage)

nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
nn.setBoundingBoxScaleFactor(0.3)
nn.setDepthLowerThreshold(100)
nn.setDepthUpperThreshold(5000)

nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(False)

# Link required inputs to the Spatial detection network
manip.out.link(nn.input)
nodes.stereo.depth.link(nn.inputDepth)

detOut = pipeline.create(dai.node.XLinkOut)
detOut.setStreamName("det_out")
nn.out.link(detOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

with dai.Device(pipeline) as device:
    replay.createQueues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det_out", maxSize=4, blocking=False)

    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.sendFrames():
        rgbFrame = cropToAspectRatio(replay.frames['color'], (300, 300))

        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame * disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        inDet = detQ.tryGet()
        if inDet is not None:
            # Display (spatial) object detections on the color frame
            for detection in inDet.detections:
                # Denormalize bounding box
                bbox = frameNorm(rgbFrame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(rgbFrame, str(label), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, "{:.2f}".format(detection.confidence * 100), (bbox[0] + 10, bbox[1] + 35),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 65),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)} mm", (bbox[0] + 10, bbox[1] + 80),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                cv2.rectangle(rgbFrame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')
