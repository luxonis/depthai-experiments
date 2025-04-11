import depthai as dai
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .annotation_helper import AnnotationHelper

class SSIM(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.passthrough_disp = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.passthrough_depth = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def setDispScaleFactor(self, dispScaleFactor):
        self.dispScaleFactor = dispScaleFactor

    def build(
        self, disp: dai.Node.Output, depth: dai.Node.Output
    ) -> "SSIM":
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
        annotation_helper = AnnotationHelper()

        annotation_helper.draw_text(
            text=f"SSIM between generated and calculated depth frame: {ssim_noise:.4f}",
            position=(0.02,0.05),
            color=(0,0,0,1),
            background_color=(1, 1, 1, 0.7),
            size=8,
        )

        annotations = annotation_helper.build(
            disp.getTimestamp(), disp.getSequenceNum()
        )

        self.output.send(annotations)
        self.passthrough_disp.send(disp)
        self.passthrough_depth.send(depth)

