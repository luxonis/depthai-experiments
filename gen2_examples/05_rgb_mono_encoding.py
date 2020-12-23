import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam = pipeline.createColorCamera()
cam.setCamId(0)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)

rgbEncoder = pipeline.createVideoEncoder()
rgbEncoder.setDefaultProfilePreset(3840, 2160, 30, depthai.VideoEncoderProperties.Profile.H264_MAIN)
cam.video.link(rgbEncoder.input)

cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

leftEncoder = pipeline.createVideoEncoder()
leftEncoder.setDefaultProfilePreset(1280, 720, 30, depthai.VideoEncoderProperties.Profile.H264_MAIN)
cam_left.out.link(leftEncoder.input)

cam_right = pipeline.createMonoCamera()
cam_right.setCamId(2)
cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

rightEncoder = pipeline.createVideoEncoder()
rightEncoder.setDefaultProfilePreset(1280, 720, 30, depthai.VideoEncoderProperties.Profile.H264_MAIN)
cam_right.out.link(rightEncoder.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName('left')
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName('right')

rgbEncoder.bitstream.link(xout_rgb.input)
leftEncoder.bitstream.link(xout_left.input)
rightEncoder.bitstream.link(xout_right.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_rgb = device.getOutputQueue("rgb")
q_left = device.getOutputQueue("left")
q_right = device.getOutputQueue("right")

with open('rgb.h264','wb') as rgbFile, open('left.h264','wb') as leftFile, open('right.h264','wb') as rightFile:
    print("press key to interrupt")
    try:
        while True:
            in_rgb = q_rgb.tryGet()
            in_left = q_left.tryGet()
            in_right = q_right.tryGet()

            if in_rgb is not None:
                in_rgb.getData().tofile(rgbFile)
            if in_left is not None:
                in_left.getData().tofile(leftFile)
            if in_right is not None:
                in_right.getData().tofile(rightFile)
    except KeyboardInterrupt:
        print("interrupted")
