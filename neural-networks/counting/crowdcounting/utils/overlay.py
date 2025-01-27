import cv2
import depthai as dai


class OverlayFrames(dai.node.HostNode):
    """A host node that receives two frames and overlays them.

    Attributes
    ----------
    output : dai.ImgFrame
        The output message for the overlayed frame.
    """
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

        self._frame_1_weight = 0.6
        self._frame_2_weight = 0.4

    def build(
        self, frame_1: dai.Node.Output, frame_2: dai.Node.Output
    ) -> "OverlayFrames":
        self.link_args(frame_1, frame_2)
        return self

    def process(self, frame_1: dai.Buffer, frame_2: dai.Buffer) -> None:
        assert isinstance(frame_1, dai.ImgFrame)
        assert isinstance(frame_2, dai.ImgFrame)

        img_1 = frame_1.getCvFrame()
        img_2 = frame_2.getCvFrame()

        if img_1.shape != img_2.shape:
            raise ValueError(
                f"ImgFrames do not have the same shape: frame_1: {img_1.shape}, frame_2: {img_2.shape}"
            )

        overlayed = cv2.addWeighted(
            img_1, self._frame_1_weight, img_2, self._frame_2_weight, 0
        )
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(overlayed, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame_1.getTimestamp())
        img_frame.setSequenceNum(frame_1.getSequenceNum())

        self.output.send(img_frame)

    def set_weigths(self, frame_1: float, frame_2: float) -> None:
        self._frame_1_weight = frame_1
        self._frame_2_weight = frame_2
