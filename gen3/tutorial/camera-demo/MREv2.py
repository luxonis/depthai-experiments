#!/usr/bin/env python3

import cv2
import depthai as dai
import datetime
from time import sleep


dataset_size = 2  # Number of image pairs
frame_interval_ms = 500


def create_pipeline():
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.XLinkIn)
    xout_left = pipeline.create(dai.node.XLinkOut)

    cam_left .setStreamName('in_left')
    xout_left.setStreamName('left')

    cam_left.out.link(xout_left.input)

    return pipeline


with dai.Device() as device:

    pipeline = create_pipeline()
    device.startPipeline(pipeline)

    q_in = device.getInputQueue('in_left')
    q_out = device.getOutputQueue('left', 8, blocking=False)

    timestamp_ms = 0
    index = 0

    while True:
        name = q_in.getName()
        path = 'dataset/' + str(index) + '/' + name + '.png'
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
        tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                    milliseconds = timestamp_ms % 1000)
        img = dai.ImgFrame()
        img.setData(data)
        img.setTimestamp(tstamp)
        img.setInstanceNum(dai.CameraBoardSocket.LEFT)
        img.setType(dai.ImgFrame.Type.RAW8)
        img.setWidth(1280)
        img.setHeight(720)

        q_in.send(img)
        
        timestamp_ms += frame_interval_ms
        index = (index + 1) % dataset_size

        # Handle output streams
        image : dai.ImgFrame = q_out.get()

        cv2.imshow("left", image.getCvFrame())
        
        if cv2.waitKey(1) == ord('q'):
            break

        sleep(1)
