import depthai as dai
import cv2


class DisplayDepth(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, depth_frame: dai.Node.Output) -> "DisplayDepth":
        self.link_args(depth_frame)
        return self

    def process(self, depth_in: dai.ImgFrame) -> None:
        frame = cv2.applyColorMap(depth_in.getCvFrame(), cv2.COLORMAP_HOT)

        depth_in.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        self.output.send(depth_in)
