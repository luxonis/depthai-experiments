import numpy as np
import cv2
import depthai as dai

CLASS_COLORS = np.array([
    [0, 0, 0]
    , [0, 255, 0]
    , [255, 0, 0]
    , [0, 0, 255]
]).astype(np.uint8)

class RoadSegmentation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "RoadSegmentation":
        self.link_args(preview, nn)

        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()
        data = nn.getTensor("L0317_ReWeight_SoftMax").squeeze()

        indices = np.argmax(data, axis=0)
        output_colors = np.take(CLASS_COLORS, indices, axis=0)

        if len(output_colors) > 0:
            cv2.addWeighted(frame, 1, cv2.resize(output_colors, frame.shape[:2][::-1]), 0.2, 0, frame)

        preview.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        self.output.send(preview)
