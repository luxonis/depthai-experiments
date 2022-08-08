"""Display depth+rgb aligned and save to point cloud file.
Adapted from https://github.com/marco-paladini/depthai-python/blob/main/examples/StereoDepth/rgb_depth_aligned.py"""
#!/usr/bin/env python3

import json
import time

import cv2
import depthai as dai
import numpy as np
import open3d as o3d

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight


# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
# widhth and height that match `monoResolution` above
width = 1280
height = 720
# max depth (only used for visualisation) in millimeters
visualisation_max_depth = 4000

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
depthOut = pipeline.create(dai.node.XLinkOut)


rgbOut.setStreamName("rgb")
queueNames.append("rgb")
depthOut.setStreamName("depth")
queueNames.append("depth")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(fps)
if downscaleColor:
    camRgb.setIspScale(2, 3)
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
stereo.depth.link(depthOut.input)
# rectified stereo
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")
stereo.rectifiedLeft.link(xoutRectifLeft.input)
stereo.rectifiedRight.link(xoutRectifRight.input)

# Connect to device and start pipeline
with device:
    serial_number = device.getMxId()
    timestamp = str(int(time.time()))
    calibFile = f"calibration_{serial_number}_{timestamp}.json"
    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)
    print("wrote", calibFile)

    M_rgb = np.array(
        calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height)
    )

    device.startPipeline(pipeline)

    frameRgb = None
    frameDepth = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar(
        "RGB Weight %", blendedWindowName, int(rgbWeight * 100), 100, updateBlendWeights
    )

    while True:
        latestPacket = {
            "rgb": None,
            "depth": None,
            "rectifiedLeft": None,
            "rectifiedRight": None,
        }

        queueEvents = device.getQueueEvents(
            ("rgb", "depth", "rectifiedLeft", "rectifiedRight")
        )
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            raw_rgb = frameRgb.copy()
            cv2.imshow(rgbWindowName, frameRgb)

        if latestPacket["rectifiedLeft"] is not None:
            raw_left = latestPacket["rectifiedLeft"].getCvFrame().copy()
            cv2.imshow("rectifiedLeft", raw_left)

        if latestPacket["rectifiedRight"] is not None:
            raw_right = latestPacket["rectifiedRight"].getCvFrame().copy()
            cv2.imshow("rectifiedRight", raw_right)

        if latestPacket["depth"] is not None:
            frameDepth = latestPacket["depth"].getFrame()
            raw_depth = frameDepth.copy().astype(np.uint16)
            frameDepth = (frameDepth * 255.0 / visualisation_max_depth).astype(np.uint8)
            frameDepth = cv2.applyColorMap(frameDepth, cv2.COLORMAP_HOT)
            frameDepth = np.ascontiguousarray(frameDepth)
            cv2.imshow(depthWindowName, frameDepth)

        # Blend when both received
        if frameRgb is not None and frameDepth is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDepth.shape) < 3:
                frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDepth, depthWeight, 0)
            cv2.imshow(blendedWindowName, blended)
            frameRgb = None
            frameDepth = None

        if cv2.waitKey(1) == ord("q"):
            # save all images
            if cv2.imwrite(f"rgb_{serial_number}_{timestamp}.png", raw_rgb):
                print("wrote", f"rgb_{serial_number}_{timestamp}.png")
            if cv2.imwrite(f"depth_{serial_number}_{timestamp}.png", raw_depth):
                print("wrote", f"depth_{serial_number}_{timestamp}.png")
            if cv2.imwrite(f"rectifiedLeft_{serial_number}_{timestamp}.png", raw_left):
                print("wrote", f"rectifiedLeft_{serial_number}_{timestamp}.png")
            if cv2.imwrite(
                f"rectifiedRight_{serial_number}_{timestamp}.png", raw_right
            ):
                print("wrote", f"rectifiedRight_{serial_number}_{timestamp}.png")
            # compute point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                image=o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)),
                    o3d.geometry.Image(raw_depth),
                ),
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=width,
                    height=height,
                    fx=M_rgb[0][0],
                    fy=M_rgb[1][1],
                    cx=M_rgb[0][2],
                    cy=M_rgb[1, 2],
                ),
            )
            # save point cloud
            if o3d.io.write_point_cloud(
                f"{serial_number}_{timestamp}.pcd", pcd, compressed=True
            ):
                print("wrote", f"{serial_number}_{timestamp}.pcd")
            # save intrinsics
            with open(
                f"{serial_number}_{timestamp}_metadata.json", "w", encoding="utf-8"
            ) as outfile:
                json.dump(
                    {
                        "serial": serial_number,
                        "fx": M_rgb[0][0],
                        "fy": M_rgb[1][1],
                        "cx": M_rgb[0][2],
                        "cy": M_rgb[1, 2],
                    },
                    outfile,
                )
            print("wrote", f"{serial_number}_{timestamp}_metadata.json")
            break
