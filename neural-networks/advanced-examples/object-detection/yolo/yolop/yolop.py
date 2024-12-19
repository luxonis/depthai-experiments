import time
import cv2
import depthai as dai
import numpy as np
from utils.functions import non_max_suppression, show_masks, show_boxes


class YoloP(dai.node.HostNode):
    def __init__(self) -> None:
        self._startTime = time.monotonic()
        self._counter = 0
        self._fps = 0
        self._confidence_threshold = 0.5
        self._iou_threshold = 0.3
        super().__init__()


    def build(self, img_frames: dai.Node.Output, nn_data: dai.Node.Output, nn_height: int, nn_width: int) -> "YoloP":
        self._nn_height = nn_height
        self._nn_width = nn_width
        self.sendProcessingToPipeline(True)
        self.link_args(img_frames, nn_data)
        return self
    

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold


    def set_iou_threshold(self, iou_threshold: float) -> None:
        self._iou_threshold = iou_threshold


    def process(self, img_frame: dai.Buffer, nn_data: dai.NNData) -> None:
        assert(isinstance(img_frame, dai.ImgFrame))
        manip_frame: np.ndarray = img_frame.getCvFrame()

        area = nn_data.getTensor("drive_area_seg").astype(np.float16).reshape((1, 2, self._nn_height, self._nn_width))
        lines = nn_data.getTensor("lane_line_seg").astype(np.float16).reshape((1, 2, self._nn_height, self._nn_width))
        dets = nn_data.getTensor("det_out").astype(np.float16).reshape((1, 6300, 6))

        boxes = np.array(non_max_suppression(dets, self._confidence_threshold, self._iou_threshold)[0])
        show_boxes(manip_frame, boxes)
        show_masks(manip_frame, area, lines)

        color_black, color_white = (0,0,0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(self._fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(manip_frame, (0,manip_frame.shape[0]-h1-6), (w1 + 2, manip_frame.shape[0]), color_white, -1)
        cv2.putText(manip_frame, label_fps, (2, manip_frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color_black)

        cv2.imshow("Predict count", manip_frame)

        self._counter += 1
        current_time = time.monotonic()
        if (current_time - self._startTime) > 1:
            self._fps = self._counter / (current_time - self._startTime)
            self._counter = 0
            self._startTime = current_time

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()