import time
import numpy as np
import cv2
import depthai as dai

from utils.utils import draw
from utils.priorbox import PriorBox


class FaceDetection(dai.node.HostNode):
    def __init__(self) -> None:
        self._counter = 0
        self._fps = 0
        self._start_time = time.time()
        super().__init__()


    def build(self, preview: dai.Node.Output, detection_network: dai.Node.Output, nn_width: int, nn_height: int, video_width: int, video_height: int) -> "FaceDetection":
        self.link_args(preview, detection_network)
        self.sendProcessingToPipeline(True)
        self._nn_width = nn_width
        self._nn_height = nn_height
        self._video_width = video_width
        self._video_height = video_height
        self._confidence_thresh = 0.6
        self._iou_thresh = 0.3
        self._keep_top_k = 750
        return self
    

    def set_start_time(self, start_time: float) -> None:
        self._start_time = start_time


    def set_confidence_thresh(self, confidence_thresh: float) -> None:
        self._confidence_thresh = confidence_thresh


    def set_iou_thresh(self, iou_thresh: float) -> None:
        self._iou_thresh = iou_thresh


    def set_keep_top_k(self, keep_top_k: int) -> None:
        self._keep_top_k = keep_top_k
        

    def process(self, message: dai.ImgFrame, detection: dai.NNData):
        frame: np.ndarray = message.getCvFrame()

        # get all layers
        conf = detection.getTensor("conf").astype(np.float16).reshape((1076, 2))
        iou = detection.getTensor("iou").astype(np.float16).reshape((1076, 1))
        loc = detection.getTensor("loc").astype(np.float16).reshape((1076, 14))

        # decode
        detections = self._get_detections(frame, conf, iou, loc)

        # NMS
        if detections.shape[0] > 0:
            # NMS from OpenCV
            bboxes = detections[:, 0:4]
            scores = detections[:, -1]

            keep_idx = cv2.dnn.NMSBoxes(
                bboxes=bboxes.tolist(),
                scores=scores.tolist(),
                score_threshold=self._confidence_thresh,
                nms_threshold=self._iou_thresh,
                eta=1,
                top_k=self._keep_top_k)  # returns [box_num, class_num]

            keep_idx = np.squeeze(keep_idx)  # [box_num, class_num] -> [box_num]
            detections = detections[keep_idx]

        if detections.shape[0] > 0:
            self._draw_detections(frame, detections)

        self._show_fps(frame)

        # show frame
        cv2.imshow("Detections", frame)

        self._update_fps_counter()

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _get_detections(self, frame: np.ndarray, conf: np.ndarray, iou: np.ndarray, loc: np.ndarray) -> np.ndarray:
        pb = PriorBox(input_shape=(self._nn_width, self._nn_height), output_shape=(frame.shape[1], frame.shape[0]))
        dets = pb.decode(loc, conf, iou, self._confidence_thresh)
        return dets
    

    def _draw_detections(self, frame, detections) -> None:
        if detections.ndim == 1:
            detections = np.expand_dims(detections, 0)

        draw(img=frame,bboxes=detections[:, :4],
             landmarks=np.reshape(detections[:, 4:14], (-1, 5, 2)),
             scores=detections[:, -1])
        
    
    def _show_fps(self, frame: np.ndarray) -> None:
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(self._fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        

    def _update_fps_counter(self) -> None:
        self._counter += 1
        if (time.time() - self._start_time) > 1:
            self._fps = self._counter / (time.time() - self._start_time)
            self._counter = 0
            self._start_time = time.time()