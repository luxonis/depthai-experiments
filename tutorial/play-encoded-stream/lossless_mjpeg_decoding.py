#!/usr/bin/env python3

### trick to solve problem with ffmpeg on linux
import cv2
import numpy as np

cv2.imshow("bugfix", np.zeros((10, 10, 3), dtype=np.uint8))
cv2.destroyWindow("bugfix")
###

import depthai as dai

from hostnodes.host_display import Display
from hostnodes.host_decode_video import DecodeVideoAv

FPS = 20.0

with dai.Pipeline() as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    video = cam_rgb.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=FPS)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setLossless(True)
    video.link(videoEnc.input)

    decoded = pipeline.create(DecodeVideoAv).build(
        enc_out=videoEnc.bitstream,
        codec="mjpeg"
    )
    decoded.set_verbose(True)
    
    color = pipeline.create(Display).build(frame=decoded.output)
    color.setName("Color")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")