#!/usr/bin/env python3

import depthai as dai

from hostnodes.host_mjpeg import DecodeFrameCV2
from hostnodes.host_display import Display

FPS = 30.0


with dai.Pipeline() as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    video = cam_rgb.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=FPS)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    video.link(videoEnc.input)

    decoded = pipeline.create(DecodeFrameCV2).build(enc_out=videoEnc.bitstream)
    
    color = pipeline.create(Display).build(frame=decoded.output)
    color.setName("Color")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")