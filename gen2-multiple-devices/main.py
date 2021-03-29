#!/usr/bin/env python3

import cv2
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

device_list = []
q_rgb_list = []

for device_info in dai.Device.getAllAvailableDevices():
    device = dai.Device(pipeline, device_info)
    device.startPipeline()
    device_list.append(device)

    # Output queue will be used to get the rgb frames from the output defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_rgb_list.append(q_rgb)

while True:
    for i, q_rgb in enumerate(q_rgb_list):
        in_rgb = q_rgb.tryGet()  # TODO wait for queue events instead
        if in_rgb is not None:
            cv2.imshow("rgb-" + str(i + 1), in_rgb.getCvFrame())

    if cv2.waitKey(1) == ord('q'):
        break

# Close the connection to all devices
for device in device_list:
    device.close()