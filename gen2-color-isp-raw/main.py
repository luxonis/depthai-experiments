#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

streams = []
# Enable one or both streams
streams.append('isp')
#streams.append('raw') # TODO: implement unpacking

display_scale = 0.3  # configurable

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setEnablePreviewStillVideoStreams(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

if 'raw' in streams:
    xout_raw = pipeline.createXLinkOut()
    xout_raw.setStreamName('raw')
    cam.raw.link(xout_raw.input)

print("===== Booting custom FW, please ignore the next failure: Failed to find device after booting")
print("===== and run a standard UVC viewer like (replace X with 0, 1, 2...) :")
print("guvcview -d /dev/videoX")
print()
device = dai.Device(pipeline, "imx283-uvc.mvcmd")
# The above will error out -- expected

device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)

capture_flag = False
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        if name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = data.getData().reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
        if name == 'raw':
            # TODO: do a proper unpack of the 10-bit pixels, this isn't correct
            shape = (height, width * 5 // 4)
            bayer = data.getData().reshape(shape).astype(np.uint8)
            bgr = bayer # Temporary, remove when unpacking is implemented
            #bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
        if capture_flag:  # Save to disk if `c` was pressed
            filename = ('capture_' + name
                        + '_' + str(width) + 'x' + str(height)
                        + '_' + str(data.getSequenceNum())
                        + '.png')
            print("Saving to file:", filename)
            bgr = np.ascontiguousarray(bgr)  # just in case
            cv2.imwrite(filename, bgr)
        # Scale for display
        display_res = (int(width * display_scale), int(height * display_scale))
        bgr = cv2.resize(bgr, display_res, interpolation=cv2.INTER_AREA)
        bgr = np.ascontiguousarray(bgr)  # just in case
        cv2.imshow(name, bgr)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(1)
    if key == ord('c'):
        capture_flag = True
    if key == ord('q'):
        break
