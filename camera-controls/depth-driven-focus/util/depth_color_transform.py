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
        self._max_disparity = 0
        self.setColormap(cv2.COLORMAP_JET)

    def setColormap(self, colormap_value: int) -> None:
        color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), colormap_value)
        color_map[0] = [0, 0, 0]
        self._colormap = color_map

    def build(self, disparity_frames: dai.Node.Output) -> "DepthColorTransform":
        self.link_args(disparity_frames)
        return self

    def setMaxDisparity(self, max_disparity: int) -> None:
        self._max_disparity = max_disparity

    def process(self, disparity_frame: dai.Buffer) -> None:
        assert isinstance(disparity_frame, dai.ImgFrame)

        frame = disparity_frame.getFrame()
        maxDisparity = max(self._max_disparity, frame.max())
        if maxDisparity == 0:
            colorizedDisparity = np.zeros(
                (frame.shape[0], frame.shape[1], 3), dtype=np.uint8
            )
        else:
            colorizedDisparity = cv2.applyColorMap(
                ((frame / maxDisparity) * 255).astype(np.uint8), self._colormap
            )
        resultFrame = dai.ImgFrame()
        resultFrame.setCvFrame(colorizedDisparity, dai.ImgFrame.Type.BGR888i)
        resultFrame.setTimestamp(disparity_frame.getTimestamp())
        self.output.send(resultFrame)
