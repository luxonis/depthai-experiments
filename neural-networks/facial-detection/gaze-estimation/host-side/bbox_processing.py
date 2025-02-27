import numpy as np
import depthai as dai
from numpy_buffer import NumpyBuffer


class BboxProcessing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output_img = dai.Node.Output(
            self,
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ],
        )
        self.output_bboxes = dai.Node.Output(self)

    def build(
        self, img_frames: dai.Node.Output, nn_data: dai.Node.Output
    ) -> "BboxProcessing":
        self.link_args(img_frames, nn_data)
        return self

    def process(self, img_frame: dai.Buffer, nn_data: dai.NNData) -> None:
        assert isinstance(img_frame, dai.ImgFrame)
        bboxes: np.ndarray = nn_data.getFirstTensor()
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]
        self.output_img.send(img_frame)
        bboxes_data = NumpyBuffer(bboxes, img_frame)
        self.output_bboxes.send(bboxes_data)
