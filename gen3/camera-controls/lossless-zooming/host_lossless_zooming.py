import cv2
import depthai as dai

class LosslessZooming(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview_1080p: dai.Node.Output, preview_full: dai.Node.Output, MJPEG: bool) -> "LosslessZooming":
        self.link_args(preview_1080p, preview_full)
        self.sendProcessingToPipeline(True)
        self.MJPEG = MJPEG
        return self

    def process(self, preview_1080p: dai.ImgFrame, preview_full: dai.ImgFrame) -> None:
        if self.MJPEG:
            # Instead of decoding, you could also save the MJPEG or stream it elsewhere. For this demo,
            # we just want to display the stream, so we need to decode it.
            frame = cv2.imdecode(preview_1080p.getData(), cv2.IMREAD_COLOR)
        else:
            yuv420Frame = preview_1080p.getData().reshape((preview_1080p.getHeight()*3 // 2, preview_1080p.getWidth()))
            frame = cv2.cvtColor(yuv420Frame, cv2.COLOR_YUV2BGR_I420)

        # Remove this line if you would like to see 1080P (not downscaled)
        frame = cv2.resize(frame, (640, 360))

        cv2.imshow('Lossless zoom 1080P', frame)
        cv2.imshow('Preview', preview_full.getCvFrame())


        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
