import time
import cv2
import depthai as dai
import numpy as np
from util.functions import non_max_suppression


class HostDecoding(dai.node.HostNode):
    def __init__(self) -> None:
        self._conf_thresh = 0.3
        self._iou_thresh = 0.4
        self._start_time = time.time()
        self._counter = 0
        self._fps = 0
        super().__init__()


    def build(self, img_output: dai.Node.Output, nn_path: str, nn_data: dai.NNData, label_map) -> "HostDecoding":
        self._nn_path = nn_path
        self._label_map = label_map
        self.link_args(img_output, nn_data)
        self.sendProcessingToPipeline(True) 
        return self


    def set_conf_thresh(self, conf_thresh: float) -> None:
        self._conf_thresh = conf_thresh


    def set_iou_thresh(self, iou_thresh: float) -> None:
        self._iou_thresh = iou_thresh


    def process(self, img_frame: dai.ImgFrame, nn_data: dai.NNData) -> None:
        frame : np.ndarray = img_frame.getCvFrame()

        output = nn_data.getTensor("output").astype(np.float16).reshape(10647, -1)

        output = np.expand_dims(output, axis = 0)

        cols = output.shape[1]
        total_classes = cols - 5

        boxes = non_max_suppression(output, conf_thres=self._conf_thresh, iou_thres=self._iou_thresh)
        boxes = np.array(boxes[0])

        if boxes is not None:
            frame = self._draw_boxes(frame, boxes, total_classes)
        cv2.putText(frame, "NN fps: {:.2f}".format(self._fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("nn_input", frame)

        self._counter += 1
        if (time.time() - self._start_time) > 1:
            self._fps = self._counter / (time.time() - self._start_time)

            self._counter = 0
            self._start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _draw_boxes(self, frame: dai.ImgFrame, boxes: np.ndarray, total_classes: int) -> np.ndarray:
        if boxes.ndim == 0:
            return frame
        else:

            # define class colors
            colors = boxes[:, 5] * (255 / total_classes)
            colors = colors.astype(np.uint8)
            colors = cv2.applyColorMap(colors, cv2.COLORMAP_HSV)
            colors = np.array(colors)

            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = int(boxes[i,0]), int(boxes[i,1]), int(boxes[i,2]), int(boxes[i,3])
                conf, cls = boxes[i, 4], int(boxes[i, 5])

                label = f"{self._label_map[cls]}: {conf:.2f}" if "default" in self._nn_path else f"Class {cls}: {conf:.2f}"
                color = colors[i, 0, :].tolist()

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # Get the width and height of label for bg square
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

                # Shows the text.
                frame = cv2.rectangle(frame, (x1, y1 - 2*h), (x1 + w, y1), color, -1)
                frame = cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        return frame