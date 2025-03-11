import depthai as dai

ENCODER_PROFILES = {
    "h264": dai.VideoEncoderProperties.Profile.H264_MAIN,
    "h265": dai.VideoEncoderProperties.Profile.H265_MAIN,
    "mjpeg": dai.VideoEncoderProperties.Profile.MJPEG,
}
