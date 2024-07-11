import depthai as dai
import cv2
from typing import Callable
import numpy as np

class QTOutput(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.stop = False

    def terminate(self):
        self.stop = True

    def build(self, preview: dai.Node.Output, stereo: dai.node.StereoDepth
              , show_callback: Callable[[np.ndarray, str], None]) -> "QTOutput":
        self.link_args(preview, stereo.disparity)
        self.sendProcessingToPipeline(True)
        self.stereo = stereo
        self.show_callback = show_callback
        return self

    def process(self, preview: dai.ImgFrame, disparity: dai.ImgFrame) -> None:
        self.show_callback(preview.getCvFrame(), "color")

        disp_frame = disparity.getCvFrame()
        disp_frame = (disp_frame * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp_frame = cv2.applyColorMap(disp_frame, cv2.COLORMAP_JET)
        self.show_callback(disp_frame, "depth")

        if self.stop:
            print("Pipeline exited.")
            self.stopPipeline()
