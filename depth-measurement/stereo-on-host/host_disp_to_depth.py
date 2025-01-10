import depthai as dai
import numpy as np
from skimage.metrics import structural_similarity as ssim


class DispToDepthControl(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def setDispScaleFactor(self, dispScaleFactor):
        self.dispScaleFactor = dispScaleFactor

    def build(
        self, disp: dai.Node.Output, depth: dai.Node.Output
    ) -> "DispToDepthControl":
        self.link_args(disp, depth)
        self.sendProcessingToPipeline(False)
        return self

    def process(self, disp: dai.ImgFrame, depth: dai.ImgFrame):
        dispFrame = np.array(disp.getFrame())
        with np.errstate(divide="ignore"):
            calcedDepth = (self.dispScaleFactor / dispFrame).astype(np.uint16)

        depthFrame = np.array(depth.getFrame())

        # Note: SSIM calculation is quite slow.
        ssim_noise = ssim(depthFrame, calcedDepth, data_range=65535)
        print(f"Similarity: {ssim_noise}")
