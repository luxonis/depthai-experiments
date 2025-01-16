#!/usr/bin/env python3

### trick to solve problem with ffmpeg on linux
import cv2
import numpy as np

cv2.imshow("bugfix", np.zeros((10, 10, 3), dtype=np.uint8))
cv2.destroyWindow("bugfix")
###

import depthai as dai  # noqa: E402

from hostnodes.host_decode_video import DecodeVideoAv  # noqa: E402
from hostnodes.host_display import Display  # noqa: E402

FPS = 30.0

with dai.Pipeline() as pipeline:
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    video = camRgb.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=FPS)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H264_MAIN)

    video.link(videoEnc.input)

    decoded = pipeline.create(DecodeVideoAv).build(
        enc_out=videoEnc.bitstream, codec="h264"
    )

    color = pipeline.create(Display).build(frame=decoded.output)
    color.setName("Color")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
