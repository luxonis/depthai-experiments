from enum import Enum
from utils.keyboard_reader import KeyboardPress
import cv2
import numpy as np
import depthai as dai
from typing import Optional, List


class MANIP_MODE(Enum):
    CROP, LETTERBOX, STRETCH = range(3)


class CombineOutputs(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.manip_mode = MANIP_MODE.CROP

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        nn_manip: dai.Node.Output,
        crop_manip: dai.Node.Output,
        letterbox_manip: dai.Node.Output,
        stretch_manip: dai.Node.Output,
        keyboard_input: Optional[dai.Node.Output] = None,
    ) -> "CombineOutputs":
        self.link_args(nn_manip, crop_manip, letterbox_manip, stretch_manip)
        self.sendProcessingToPipeline(True)
        self.keyboard_input_q = keyboard_input.createOutputQueue()
        return self

    def process(
        self,
        nn_manip: dai.ImgFrame,
        crop_manip: dai.ImgFrame,
        letterbox_manip: dai.ImgFrame,
        stretch_manip: dai.ImgFrame,
    ) -> None:
        crop_frame = crop_manip.getCvFrame()
        cv2.putText(
            crop_frame,
            "'a' to select crop",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        letterbox_frame = letterbox_manip.getCvFrame()
        cv2.putText(
            letterbox_frame,
            "'s' to select letterbox",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        stretch_frame = stretch_manip.getCvFrame()
        cv2.putText(
            stretch_frame,
            "'d' to select stretch",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        detections_frame = nn_manip.getCvFrame()
        self.process_keyboard_input()

        if self.manip_mode == MANIP_MODE.CROP:
            combined_frame = np.concatenate(
                (detections_frame, letterbox_frame, stretch_frame), axis=1
            )
        elif self.manip_mode == MANIP_MODE.LETTERBOX:
            combined_frame = np.concatenate(
                (crop_frame, detections_frame, stretch_frame), axis=1
            )
        else:
            combined_frame = np.concatenate(
                (crop_frame, letterbox_frame, detections_frame), axis=1
            )

        self.output.send(self.get_frame(combined_frame, dai.ImgFrame.Type.BGR888p))

    def get_frame(self, frame, type):
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        return img_frame

    def process_keyboard_input(self) -> None:
        # config = dai.ImageManipConfig()

        try:
            key_presses: List[KeyboardPress] = self.keyboard_input_q.tryGetAll()
        except dai.MessageQueue.QueueException:
            return

        if key_presses:
            for key_press in key_presses:
                if key_press.key == ord("q"):
                    print("Pipeline exited.")
                    self.stopPipeline()
                elif key_press.key == ord("a"):
                    # config.setKeepAspectRatio(True)
                    # config.setResize(300, 300)
                    # # self._config.send(config)
                    self.manip_mode = MANIP_MODE.CROP
                    print("Switched to crop mode.")
                elif key_press.key == ord("s"):
                    # config.setKeepAspectRatio(True)
                    # config.setResizeThumbnail(300, 300)
                    # # self._config.send(config)
                    self.manip_mode = MANIP_MODE.LETTERBOX
                    print("Switched to letterbox mode.")
                elif key_press.key == ord("d"):
                    # config.setKeepAspectRatio(False)
                    # config.setResize(300, 300)
                    # # self._config.send(config)
                    self.manip_mode = MANIP_MODE.STRETCH
                    print("Switched to stretch mode.")
