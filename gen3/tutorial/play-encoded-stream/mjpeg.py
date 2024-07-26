#!/usr/bin/env python3

import depthai as dai
import cv2


class PlayEncodedVideo(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, enc_out) -> "PlayEncodedVideo":
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, enc_vid) -> None:
        # Receive encoded frame
        imgFrame: dai.ImgFrame = enc_vid
        # Decode the MJPEG frame
        frame = cv2.imdecode(imgFrame.getData(), cv2.IMREAD_COLOR)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
    camRgb.video.link(videoEnc.input)

    pipeline.create(PlayEncodedVideo).build(
        enc_out=videoEnc.bitstream
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
