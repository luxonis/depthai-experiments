import depthai as dai
import cv2
import numpy as np

class DisplayDepth(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, depth_frame: dai.Node.Output, max_disparity: int) -> "DisplayDepth":
        self.disp_multiplier = 255 / max_disparity

        self.link_args(depth_frame)
        return self

    def process(self, depth_in: dai.ImgFrame) -> None:
        frame = (depth_in.getFrame() * self.disp_multiplier).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

        depth_in.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        self.output.send(depth_in)
