import depthai as dai

from utils import RotatedRectBuffer

class ProcessDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.passthrough = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output_rect = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)])
        self.output_config = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)])

    def build(self, frame: dai.Node.Output, detections: dai.Node.Output) -> "ProcessDetections":
        self.link_args(frame, detections)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        # Casting values that were sent in this exact format
        rotated_rectangles = [((d.xmin, d.ymin), (d.xmax, d.ymax), d.confidence) for d in detections.detections]

        for idx, rotated_rect in enumerate(rotated_rectangles):
            rr = dai.RotatedRect()
            rr.center.x = rotated_rect[0][0]
            rr.center.y = rotated_rect[0][1]
            rr.size.width = rotated_rect[1][0]
            rr.size.height = rotated_rect[1][1]
            rr.angle = rotated_rect[2]

            cfg = dai.ImageManipConfig()
            cfg.setCropRotatedRect(rr, False)
            cfg.setResize(120, 32)
            cfg.setTimestamp(frame.getTimestamp())
            cfg.setTimestampDevice(frame.getTimestampDevice())

            rr_buffer = RotatedRectBuffer()
            rr_buffer.set_rect(rr)
            rr_buffer.setTimestamp(frame.getTimestamp())
            rr_buffer.setTimestampDevice(frame.getTimestampDevice())

            # Send outputs to device
            if idx == 0:
                self.passthrough.send(frame)
                cfg.setReusePreviousImage(False)
            else:
                cfg.setReusePreviousImage(True)
            self.output_rect.send(rr_buffer)
            self.output_config.send(cfg)

