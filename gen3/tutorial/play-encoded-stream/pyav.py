#!/usr/bin/env python3

import depthai as dai
import av
import cv2


class PlayEncodedVideo(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, enc_out) -> "PlayEncodedVideo":
        self.codec = av.CodecContext.create("h264", "r")
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        self.inputs['enc_vid'].setMaxSize(30)
        self.inputs['enc_vid'].setBlocking(True)
        return self
    

    def process(self, enc_vid) -> None:
        data = enc_vid.getData()
        packets = self.codec.parse(data)

        for packet in packets:
            frames = self.codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format='bgr24')
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
    camRgb.video.link(videoEnc.input)
    
    pipeline.create(PlayEncodedVideo).build(
        enc_out=videoEnc.bitstream
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
