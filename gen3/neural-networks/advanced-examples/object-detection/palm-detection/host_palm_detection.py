import depthai as dai
import cv2


class PalmDetection(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, preview: dai.Node.Output, palm_detections: dai.Node.Output) -> "PalmDetection":
        self.link_args(preview, palm_detections)
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        return self


    def process(self, preview: dai.ImgFrame, palm_detections: dai.Buffer) -> None:
        frame = preview.getCvFrame()
        assert(isinstance(palm_detections, dai.ImgDetections))
        for detection in palm_detections.detections:
            cv2.rectangle(
                frame,
                (int(detection.xmin * frame.shape[1]), int(detection.ymin * frame.shape[0])),
                (int(detection.xmax * frame.shape[1]), int(detection.ymax * frame.shape[0])),
                (255, 0, 0),
                2,
            )
            
        preview.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        self.output.send(preview)
