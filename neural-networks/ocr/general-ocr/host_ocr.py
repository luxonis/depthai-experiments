import depthai as dai
import cv2
import numpy as np

from utils import CTCCodec
from detected_recognitions import DetectedRecognitions

CODEC = CTCCodec("0123456789abcdefghijklmnopqrstuvwxyz#")


class OCR(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        preview: dai.Node.Output,
        manips: dai.Node.Output,
        recognitions: dai.Node.Output,
    ) -> "OCR":
        self.link_args(preview, manips, recognitions)
        self.sendProcessingToPipeline(True)
        return self

    # manips and recognitions are actually type DetectedRecognitions
    def process(
        self, preview: dai.ImgFrame, manips: dai.Buffer, recognitions: dai.Buffer
    ) -> None:
        assert isinstance(manips, DetectedRecognitions)
        assert isinstance(recognitions, DetectedRecognitions)

        # Make a stacked frame out of text recognition
        stacked_frame = None
        text_placeholder = np.zeros((48, 400, 3), np.uint8)

        if manips.data is not None and recognitions.data is not None:
            for manip, recognition in zip(manips.data, recognitions.data):
                if len(recognition.classes) == 0:
                    decoded_text = "*no recognition*"
                else:
                    decoded_text = " ".join(recognition.classes)

                # print(decoded_text, "decoded text")
                text_frame = text_placeholder.copy()
                cv2.putText(
                    text_frame,
                    decoded_text,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                stack_layer = np.hstack((manip.getCvFrame(), text_frame))
                if stacked_frame is None:
                    stacked_frame = stack_layer
                else:
                    stacked_frame = np.vstack((stacked_frame, stack_layer))

        if stacked_frame is not None:
            cv2.imshow("Recognition", stacked_frame)
        # cv2.imshow("Preview", preview.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
