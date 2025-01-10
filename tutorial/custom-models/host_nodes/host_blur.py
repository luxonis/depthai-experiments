import depthai as dai
import numpy as np

BLUR_SHAPE = 300


class ReshapeNNOutputBlur(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.shape = (3, BLUR_SHAPE, BLUR_SHAPE)

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(self, nn_out: dai.Node.Output) -> "ReshapeNNOutputBlur":
        self.link_args(nn_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, nn_det: dai.NNData) -> None:
        frame = self.get_frame(nn_det)

        img = dai.ImgFrame()
        img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)

        self.output.send(img)

    def get_frame(self, imfFrame):
        return (
            np.array(imfFrame.getData())
            .view(np.float16)
            .reshape(self.shape)
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
