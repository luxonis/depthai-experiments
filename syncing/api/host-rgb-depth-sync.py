#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


msgs = dict()

def add_msg(msg, name, seq = None):
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2: # rgb + depth
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs # Returned synced msgs
    return None

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
disparityOut.setStreamName("disp")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(2, 3)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    while True:
        for name in ['rgb', 'disp']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()
        if synced:
            frameRgb = synced["rgb"].getCvFrame()

            frameDisp = synced["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)

            frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_TURBO)
            frameDisp = np.ascontiguousarray(frameDisp)

            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
            cv2.imshow(blendedWindowName, blended)
            frameRgb = None
            frameDisp = None

        if cv2.waitKey(1) == ord('q'):
            break
