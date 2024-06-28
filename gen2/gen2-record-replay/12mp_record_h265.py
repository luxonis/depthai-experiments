#!/usr/bin/env python3

import depthai as dai
import av
from fractions import Fraction

# Create pipeline
pipeline = dai.Pipeline()
# VideoEncoder H265/H265 limitation; max 250MP/sec, which is about 20FPS @ 12MP
fps = 20

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(fps)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # (4056, 3040)

"""
We have to use ImageManip, as ColorCamera.video can output up to 4K.
Workaround is to use ColorCamera.isp, and convert it to NV12
"""
imageManip = pipeline.create(dai.node.ImageManip)
# YUV420 -> NV12 (required by VideoEncoder)
imageManip.initialConfig.setFrameType(dai.RawImgFrame.Type.NV12)
# Width must be multiple of 32, height multiple of 8 for H26x encoder
imageManip.initialConfig.setResize(4032, 3040)
imageManip.setMaxOutputFrameSize(18495360)
camRgb.isp.link(imageManip.inputImage)

# Properties
videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.H265_MAIN)
imageManip.out.link(videoEnc.input)

# Linking
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('bitstream')
videoEnc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device, av.open('video.mp4', mode='w') as file:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="bitstream", maxSize=30, blocking=True)

    codec = av.CodecContext.create('hevc', 'w')
    stream = file.add_stream('hevc')
    stream.width = 4032
    stream.height = 3040
    stream.time_base = Fraction(1, 1000 * 1000)

    start_ts = None

    print('Press CTRL+C to stop recording... Use VNC to view video.')

    while True:
        frame: dai.ImgFrame = q.get()  # Blocking call, will wait until a new data has arrived

        if start_ts is None:
            start_ts = frame.getTimestampDevice()

        packet = av.Packet(frame.getData())

        ts = int((frame.getTimestampDevice() - start_ts).total_seconds() * 1e6)  # To microsec
        packet.dts = ts + 1  # +1 to avoid zero dts
        packet.pts = ts + 1
        packet.stream = stream
        file.mux_one(packet)  # Mux the Packet into container

