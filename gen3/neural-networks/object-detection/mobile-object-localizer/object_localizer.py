import time
import cv2
import numpy as np
import depthai as dai


class ObjectLocalizer(dai.node.HostNode):
    def __init__(self) -> None:
        np.random.seed(0)
        self._colors_full = np.random.randint(255, size=(100, 3), dtype=int)
        self._threshold = 0.7
        self._counter = 0
        self._fps = 0
        self._start_time = time.time()
        super().__init__()


    def build(self, cam: dai.Node.Output, nn: dai.Node.Output, manip: dai.Node.Output) -> "ObjectLocalizer":
        self.link_args(cam, nn, manip)
        self.sendProcessingToPipeline(True)
        return self


    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold


    def process(self, cam: dai.ImgFrame, nn: dai.ImgDetections, manip: dai.ImgFrame) -> None:
        frame: np.ndarray = cam.getCvFrame()
        frame_manip = manip.getCvFrame()

        detections = nn.detections

        for i, detection in enumerate(detections):
            if detection.confidence < self._threshold: continue

            color = (int(self._colors_full[i, 0]), int(self._colors_full[i, 1]), int(self._colors_full[i, 2]))

            self._draw_boxes(frame, detection, color, detection.confidence)
            self._draw_boxes(frame_manip, detection, color, detection.confidence)

            cv2.imshow("Localizer", frame)
            cv2.imshow("Manip + NN", frame_manip)

            self._show_fps(frame)
            self._update_fps_counter()

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _frame_norm(self, frame: np.ndarray, bbox: tuple):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
    
    
    def _draw_boxes(self, frame: np.ndarray, detection: np.ndarray, color, scores) -> None:
        color_black = (0, 0, 0)

        bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, str(scores), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_black) 


    def _show_fps(self, frame: np.ndarray) -> None:
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(self._fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)

        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        
    
    def _update_fps_counter(self):
        self._counter += 1
        if (time.time() - self._start_time) > 1:
            self._fps = self._counter / (time.time() - self._start_time)

            self._counter = 0
            self._start_time = time.time()