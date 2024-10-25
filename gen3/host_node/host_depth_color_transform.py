import cv2
import depthai as dai
import numpy as np


class DepthColorTransform(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.colormap = cv2.COLORMAP_HOT

    def build(
        self, disparity_frames: dai.Node.Output, max_disparity: int
    ) -> "DepthColorTransform":
        self.disp_multiplier = 255 / max_disparity
        self.link_args(disparity_frames)
        return self

    def setColormap(self, colormap: int) -> None:
        self.colormap = colormap

    def process(self, disparity_frame: dai.Buffer) -> None:
        assert isinstance(disparity_frame, dai.ImgFrame)

        frame = (disparity_frame.getFrame() * self.disp_multiplier).astype(np.uint8)
        frame = cv2.applyColorMap(frame, self.colormap)

        disparity_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        self.output.send(disparity_frame)
