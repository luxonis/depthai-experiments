import depthai as dai
import numpy as np

CONCAT_SHAPE = 300


class ReshapeNNOutputConcat(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.shape = (3, CONCAT_SHAPE, CONCAT_SHAPE * 3)

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(self, nn_out: dai.Node.Output) -> "ReshapeNNOutputConcat":
        self.link_args(nn_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, nn_data: dai.NNData):
        inNn = np.array(nn_data.getData())
        frame = (
            inNn.view(np.float16)
            .reshape(self.shape)
            .transpose(1, 2, 0)
            .astype(np.uint8)
            .copy()
        )

        img = dai.ImgFrame()
        img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)

        self.output.send(img)
