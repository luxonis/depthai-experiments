from enum import Enum, auto
from math import ceil, floor

import cv2
import depthai as dai
import numpy as np


class FrameStacker(dai.node.HostNode):
    class StackingMode(Enum):
        HORIZONTAL = auto()
        VERTICAL = auto()

    class ResizeMode(Enum):
        STRETCH = auto()
        PAD = auto()

    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self._stacking_mode = self.StackingMode.HORIZONTAL
        self._resize_mode = self.ResizeMode.STRETCH

    def set_stacking_mode(self, stacking_mode: StackingMode) -> None:
        self._stacking_mode = stacking_mode

    def set_resize_mode(self, resize_mode: ResizeMode) -> None:
        self._resize_mode = resize_mode

    def build(
        self, frame_1: dai.Node.Output, frame_2: dai.Node.Output
    ) -> "FrameStacker":
        self.link_args(frame_1, frame_2)
        return self

    def process(self, frame_1: dai.ImgFrame, frame_2: dai.ImgFrame) -> None:
        all_frames: list[dai.ImgFrame] = [frame_1, frame_2]

        images = [f.getCvFrame() for f in all_frames]
        resized_images = self._match_image_sizes(images)
        stacked_image = self._stack_images(resized_images)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(stacked_image, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame_1.getTimestamp())
        img_frame.setSequenceNum(frame_1.getSequenceNum())

        self.output.send(img_frame)

    def _match_image_sizes(self, images: list[np.ndarray]) -> list[np.ndarray]:
        max_x_size = max(f.shape[1] for f in images)
        max_y_size = max(f.shape[0] for f in images)
        resized_images = []
        for img in images:
            if self._stacking_mode == self.StackingMode.HORIZONTAL:
                new_shape = (img.shape[1], max_y_size)
            elif self._stacking_mode == self.StackingMode.VERTICAL:
                new_shape = (max_x_size, img.shape[0])
            else:
                raise ValueError("Invalid StackingMode")

            if self._resize_mode == self.ResizeMode.PAD:
                new_img = self._pad_image(img, new_shape)
            elif self._resize_mode == self.ResizeMode.STRETCH:
                new_img = self._resize_image(img, new_shape)
            else:
                raise ValueError("Invalid ResizeMode")
            resized_images.append(new_img)
        return resized_images

    def _stack_images(self, images: list[np.ndarray]) -> np.ndarray:
        if self._stacking_mode == self.StackingMode.HORIZONTAL:
            return np.hstack(images)
        elif self._stacking_mode == self.StackingMode.VERTICAL:
            return np.vstack(images)
        else:
            raise ValueError("Invalid StackingMode")

    def _resize_image(
        self, image: np.ndarray, target_size: tuple[int, int]
    ) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def _pad_image(self, image: np.ndarray, target_size: tuple[int, int]):
        top_padding = floor((target_size[1] - image.shape[0]) / 2.0)
        bottom_padding = ceil((target_size[1] - image.shape[0]) / 2.0)
        left_padding = floor((target_size[0] - image.shape[1]) / 2.0)
        right_padding = ceil((target_size[0] - image.shape[1]) / 2.0)

        return cv2.copyMakeBorder(
            image,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=0,
        )
