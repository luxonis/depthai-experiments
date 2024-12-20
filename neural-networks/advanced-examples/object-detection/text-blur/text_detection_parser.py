import depthai as dai
import numpy as np


class TextDetectionParser(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self._conf_threshold = 0.5

    def build(
        self, nn: dai.Node.Output, nn_size: tuple[int, int]
    ) -> "TextDetectionParser":
        self.link_args(nn)
        self._nn_size = nn_size
        return self

    def process(self, nn_data: dai.NNData) -> None:
        pred = nn_data.getTensor("output").reshape(self._nn_size)
        mask = (pred > self._conf_threshold).astype(np.uint8)

        img_frame = dai.ImgFrame()
        img_frame.setFrame(mask)
        img_frame.setType(dai.ImgFrame.Type.RAW8)
        img_frame.setWidth(self._nn_size[0])
        img_frame.setHeight(self._nn_size[1])
        img_frame.setTimestamp(nn_data.getTimestamp())
        img_frame.setSequenceNum(nn_data.getSequenceNum())
        self.output.send(img_frame)

    def setConfidenceThreshold(self, threshold: float) -> None:
        self._conf_threshold = threshold
