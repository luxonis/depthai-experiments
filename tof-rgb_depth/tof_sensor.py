#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from matplotlib import pyplot as plt, cm
pipeline = dai.Pipeline()

cam_a = pipeline.createColorCamera()
# We assume the ToF camera sensor is on port CAM_A
cam_a.setBoardSocket(dai.CameraBoardSocket.CAM_A)

tof = pipeline.create(dai.node.ToF)
xout = pipeline.createXLinkOut()
xout.setStreamName("depth")
# Configure the ToF node
tofConfig = tof.initialConfig.get()
# tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MIN
tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MAX
tofConfig.depthParams.avgPhaseShuffle = False
tofConfig.depthParams.minimumAmplitude = 3.0
tof.initialConfig.set(tofConfig)
# Link the ToF sensor to the ToF node
cam_a.raw.link(tof.input)

tof.depth.link(xout.input)

def get_rgb_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = depth_map[y, x]*0.1
        print(f"RGB Value at ({x}, {y}): {pixel_value}cm")

def draw_roi(event, x, y, flags, param):
    global roi, cropping, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = [x, y]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        roi.append(x)
        roi.append(y)
        cropping = False

def _mask_frame( frame: np.ndarray, roi) -> np.ndarray:
    if roi is not None:
        assert (len(roi) == 4)
        x1, y1, x2, y2 = roi
        roi_sorted = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        x1, y1, x2, y2 = roi_sorted
        mask = np.zeros_like(frame)
        mask[y1:y2, x1:x2] = 1
    else:
        mask = np.ones_like(frame)
    frame = frame * mask
    return frame
roi = []
cropping = False
distance = 30
cv2.namedWindow('Image')
# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras:', device.getConnectedCameraFeatures())
    q = device.getOutputQueue(name="depth")
    print(q)

    while True:
        imgFrame = q.get()  
        # blocking call, will wait until a new data has arrived
        depth_map = imgFrame.getFrame()
        # Colorize the depth frame to jet colormap
        depth_downscaled = depth_map[::4]
        non_zero_depth = depth_downscaled[depth_downscaled != 0] # Remove invalid depth values
        if len(non_zero_depth) == 0:
            min_depth, max_depth = 0, 0
        else:
            min_depth = np.percentile(non_zero_depth, 3)
            max_depth = np.percentile(non_zero_depth, 97)
        depth_colorized = np.interp(depth_map, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        if len(roi) == 4 and abs(roi[0]-roi[2])>10 and abs(roi[1]-roi[3])>10:
            cv2.rectangle(depth_colorized, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), 5)
            depth_map2 = cv2.rectangle(depth_map, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), 5)
            depth_np = _mask_frame(depth_map, roi=roi)
            print(depth_np[roi[0]:roi[2], roi[1]:roi[3]])
            x1, y1, x2, y2 = roi
            print(f"RGB Value of ROI is: {np.mean(depth_np[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2)])}m, error {np.mean(depth_np)-distance}m")
            min_depth = np.percentile(depth_np, 3)
            max_depth = np.percentile(depth_np, 97)
            depth_color = np.interp(depth_np, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)
            cv2.imshow("Cropped depth", depth_np-distance)
        cv2.setMouseCallback('Image', draw_roi)
        key = cv2.waitKey(1)

        depth_colorized = cv2.applyColorMap(depth_colorized, cv2.COLORMAP_JET)
        cv2.imshow('Image', depth_colorized)

        if key == ord('q'):
            break
plt.title(f"Depth map of ToF sensor, distance {distance}cm")
plt.imshow(depth_map*0.1, cmap=plt.cm.jet)
cbar = plt.colorbar()
cbar.set_label("Depth[cm]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.title(f"Error map of ToF sensor, distance {distance}cm")
plt.imshow((depth_map*0.1-distance)/distance*100, cmap=plt.cm.jet)
cbar = plt.colorbar()
cbar.set_label("Error[%]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()