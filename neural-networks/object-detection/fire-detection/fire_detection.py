import numpy as np
import cv2
import depthai as dai


class FireDetection(dai.node.HostNode):
    LABELS = ["fire", "normal", "smoke"]

    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self._font_scale = 1
        self._line_type = 0
        self._text_color = (0, 0, 255)

    def set_font_scale(self, font_scale: float) -> None:
        self._font_scale = font_scale

    def set_line_type(self, line_type: int) -> None:
        self._line_type = line_type

    def set_text_color(self, color: tuple[int, int, int]) -> None:
        self._text_color = color

    def build(
        self, img_frames: dai.Node.Output, nn_data: dai.NNData
    ) -> "FireDetection":
        self.link_args(img_frames, nn_data)
        return self

    def process(self, img_frame: dai.Buffer, nn_data: dai.NNData) -> None:
        assert isinstance(img_frame, dai.ImgFrame)
        frame: np.ndarray = img_frame.getCvFrame()
        results = nn_data.getTensor("final_result").flatten()
        i = int(np.argmax(results))
        label = self.LABELS[i]
        if label == "normal":
            self.output.send(img_frame)
            return
        elif results[i] > 0.5:
            self._put_text(frame, f"{label}:{results[i]:.2f}", (10, 25))

        output_frame = dai.ImgFrame()
        output_frame.setType(dai.ImgFrame.Type.BGR888i)
        output_frame.setFrame(frame)
        output_frame.setWidth(img_frame.getWidth())
        output_frame.setHeight(img_frame.getHeight())
        output_frame.setTimestamp(img_frame.getTimestamp())
        output_frame.setTimestampDevice(img_frame.getTimestampDevice())
        output_frame.setInstanceNum(img_frame.getInstanceNum())

        self.output.send(output_frame)

    def _put_text(self, frame: np.ndarray, text: str, dot: tuple[int, int]) -> None:
        dot = tuple(dot[:2])
        cv2.putText(
            img=frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=self._font_scale,
            color=self._text_color,
            lineType=self._line_type,
        )
