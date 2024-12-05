import numpy as np
import depthai as dai
import cv2

from utils import BeamSearchDecoder
from scipy.special import softmax

class SevenSegmentRecognition(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.codec = BeamSearchDecoder("1234567890.", beam_len=30)

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "SevenSegmentRecognition":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()

        data = nn.getTensor("845").flatten().reshape(24, 1, 12)
        data = np.transpose(data, [1, 0, 2])
        classes_softmax = softmax(data, 2)[0]
        predictions = self.codec.decode(classes_softmax)

        cv2.putText(frame, predictions, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), thickness=3)
        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

