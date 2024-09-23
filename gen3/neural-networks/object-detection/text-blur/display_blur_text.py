import numpy as np
import cv2
import depthai as dai
from utils.utils import get_boxes


class DisplayBlurText(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._bbox_threshold = 0.2
        self._bitmap_threshold = 0.01
        self._min_size = 1
        self._max_candidates = 75
        self._draw_bboxes = False
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output_preds = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])


    def build(self, camOut : dai.Node.Output, nnOut : dai.Node.Output, nn_size: tuple[int, int]) -> "DisplayBlurText":
        self.link_args(camOut, nnOut)
        self._nn_size = nn_size
        return self
    

    def process(self, in_frame : dai.ImgFrame, nn_data : dai.NNData) -> None:
        frame = in_frame.getCvFrame()

        pred = nn_data.getTensor("output").reshape(self._nn_size)

        # Output mask
        result_pred = dai.ImgFrame()
        result_pred.setCvFrame((pred * 255).astype(np.uint8), dai.ImgFrame.Type.GRAY8)
        result_pred.setTimestamp(in_frame.getTimestamp())
        self.output_preds.send(result_pred)

        # Decode
        boxes, scores = get_boxes(pred, self._bitmap_threshold, self._bbox_threshold, self._min_size, self._max_candidates)

        # Blur image
        blur = cv2.GaussianBlur(frame, (49, 49), 30)

        for i, box in enumerate(boxes):
            if self._draw_bboxes:
                # Draw boxes
                cv2.rectangle(frame, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), (255, 0, 0), 1)
                cv2.putText(frame, f"Score: {scores[i]:.2f}", (box[0,0], box[0,1]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0))

            # Blur boxes
            x1, y1, x2, y2 = box[0, 0] - 5, box[0, 1] - 5, box[2, 0] + 5, box[2, 1] + 5
            x1, x2 = np.clip([x1, x2], 0, frame.shape[1])
            y1, y2 = np.clip([y1, y2], 0, frame.shape[0])
            frame[y1:y2, x1:x2] = blur[y1:y2, x1:x2]

        output_frame = dai.ImgFrame()
        output_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
        output_frame.setTimestamp(in_frame.getTimestamp())
        self.output.send(output_frame)


    def setBboxThreshold(self, threshold: float) -> None:
        self._bbox_threshold = threshold


    def setBitmapThreshold(self, threshold: float) -> None:
        self._bitmap_threshold = threshold


    def setMinSize(self, min_size: int) -> None:
        self._min_size = min_size


    def setMaxCandidates(self, max_candidates: int) -> None:
        self._max_candidates = max_candidates


    def setBboxes(self, draw_bboxes: bool) -> None:
        self._draw_bboxes = draw_bboxes
