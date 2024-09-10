import depthai as dai
from typing import Callable
import numpy as np

class QTOutput(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.stop = False

    def terminate(self):
        self.stop = True

    def build(self, preview: dai.Node.Output, disparity: dai.Node.Output, 
              show_callback: Callable[[np.ndarray, str], None]) -> "QTOutput":
        
        self.link_args(preview, disparity)
        self.sendProcessingToPipeline(True)
        self.show_callback = show_callback
        return self

    def process(self, preview: dai.ImgFrame, disparity: dai.ImgFrame) -> None:
        self.show_callback(preview.getCvFrame(), "color")
        self.show_callback(disparity.getCvFrame(), "depth")

        if self.stop:
            print("Pipeline exited.")
            self.stopPipeline()