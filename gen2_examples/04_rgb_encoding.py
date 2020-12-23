import cv2
import numpy as np
import depthai

pipeline = depthai.Pipeline()

cam = pipeline.createColorCamera()
cam.setCamId(0)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)

videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(3840, 2160, 30, depthai.VideoEncoderProperties.Profile.H264_MAIN)
cam.video.link(videoEncoder.input)

videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h264')
videoEncoder.bitstream.link(videoOut.input)

device = depthai.Device(pipeline)
device.startPipeline()

q = device.getOutputQueue('h264')

with open('video.h264','wb') as videoFile:
    print("Press xxx to stop encoding...")
    try:
        while True:
            h264Packet = q.get()
            h264Packet.getData().tofile(videoFile)
    except KeyboardInterrupt:
        print("inter")
