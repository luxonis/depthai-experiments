import time
import cv2
import depthai as dai
import numpy as np


class Preprocess(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input = dai.Node.Input(self, blocking=False, queueSize=4)
        self.output = dai.Node.Output(self)


    def build(self, mean: tuple[int, int, int], std: tuple[int, int, int], shape: int) -> "Preprocess":
        self._mean = mean
        self._std = std
        self._shape = shape
        return self


    def run(self):
        while self.isRunning():
            img_frame: dai.ImgFrame = self.input.get()
            frame = img_frame.getCvFrame()

            image, ratio = self._preproc(frame, (self._shape, self._shape), self._mean, self._std)
            # NOTE: The model expects an FP16 input image, but ImgFrame accepts a list of ints only. I work around this by
            # spreading the FP16 across two ints
            image = list(image.tobytes())

            dai_frame = dai.ImgFrame()
            dai_frame.setHeight(self._shape)
            dai_frame.setWidth(self._shape)
            dai_frame.setData(image)
            dai_frame.setTimestamp(img_frame.getTimestamp())
            dai_frame.setTimestampDevice(img_frame.getTimestampDevice())
            dai_frame.setSequenceNum(img_frame.getSequenceNum())
            
            self.output.send(dai_frame)


    def _preproc(self, image: np.ndarray, input_size: tuple, mean: tuple, std: tuple, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float16)
        return padded_img, r
