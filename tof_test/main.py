#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

print("depthai version:", dai.__version__)

pipeline = dai.Pipeline()
streams = []

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout_raw = pipeline.createXLinkOut()
xout_raw.setStreamName('raw')
cam.raw.link(xout_raw.input)
streams.append('raw')

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=False)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (300, 960))

capture_flag = False
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        frame = data.getData()
        capture_file_info_str = ('capture_' + name
                                 + '_' + str(width) + 'x' + str(height)
                                 + '_' + str(data.getSequenceNum())
                                )
        if name == 'raw':
            # Next line is a workaround for improperly set size (1080p)
            frame = np.resize(frame, height*width*2)
            if capture_flag:
                # Save to capture file the unpacked data, right-justified
                filename = capture_file_info_str + '_12bit.bw'
                print("Saving to file:", filename)
                frame.tofile(filename)
            shape = (height, width)
            # Left-justify for better visualization
            frame = (frame.view(np.uint16) * 16).reshape(shape)
        if capture_flag:  # Save to disk if `c` was pressed
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            frame = np.ascontiguousarray(frame)  # just in case
            cv2.imwrite(filename, frame)
        frame = np.ascontiguousarray(frame)  # just in case
        cv2.imshow(name, frame)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(1)
    if key == ord('c'):
        capture_flag = True
    elif key == ord('q'):
        break
