#!/usr/bin/env python3

import depthai as dai
import av
import cv2
import time

class DecodeFrames(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, enc_out : dai.Node.Output) -> "DecodeFrames":
        self.codec = av.CodecContext.create("mjpeg", "r")
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        self.inputs['enc_vid'].setMaxSize(1)
        self.inputs['enc_vid'].setBlocking(False)
        return self
    

    def process(self, enc_vid) -> None:
        data = enc_vid.getData()
        start = time.perf_counter()
        packets = self.codec.parse(data)
        for packet in packets:
            frames = self.codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format='bgr24')
                print(f"AV decode frame in {(time.perf_counter() - start) * 1000} milliseconds")
                cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(8)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setLossless(True)
    camRgb.video.link(videoEnc.input)

    pipeline.create(DecodeFrames).build(
        enc_out=videoEnc.bitstream
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
