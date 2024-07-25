#!/usr/bin/env python3

import cv2
import depthai as dai
import datetime
from time import sleep

dataset_size = 2
frame_interval = 500

with dai.Pipeline() as pipeline:
    image_manip_left = pipeline.create(dai.node.ImageManip)

    img_l_in = image_manip_left.inputImage.createInputQueue()

    left_q = image_manip_left.out.createOutputQueue()

    index = 0
    timestamp_ms = 0

    pipeline.start()

    while pipeline.isRunning():

        path = 'dataset/' + str(index) + '/' + 'in_left' + '.png'
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
        tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                    milliseconds = timestamp_ms % 1000)
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.RAW8)
        img.setFrame(data)
        img.setInstanceNum(dai.CameraBoardSocket.CAM_B)
        img.setTimestamp(tstamp)
        img.setWidth(1280)
        img.setHeight(720)

        print("img", img.getFrame()) # image

        img_l_in.send(img)

        timestamp_ms += frame_interval
        index = (index + 1) % dataset_size
        
        left : dai.ImgFrame = left_q.tryGet()

        if left != None:
            print("left", left.getCvFrame()) # zeroes
            cv2.imshow("left", left.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
        
        sleep(1)