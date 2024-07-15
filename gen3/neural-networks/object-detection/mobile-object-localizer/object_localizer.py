import time
import cv2
import numpy as np
import depthai as dai


class ObjectLocalizer(dai.node.HostNode):
    def __init__(self) -> None:
        np.random.seed(0)
        self._colors_full = np.random.randint(255, size=(100, 3), dtype=int)
        self._threshold = 0.2
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


    def process(self, cam: dai.ImgFrame, nn: dai.NNData, manip: dai.ImgFrame) -> None:
        frame: np.ndarray = cam.getCvFrame()
        frame_manip = manip.getCvFrame()
        frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)

        detection_boxes = nn.getTensor("ExpandDims").astype(np.float16).reshape((100, 4))
        detection_scores = nn.getTensor("ExpandDims_2").astype(np.float16).reshape((100,))

        mask = detection_scores >= self._threshold
        boxes = detection_boxes[mask]
        colors = self._colors_full[mask]
        scores = detection_scores[mask]

        self._plot_boxes(frame, boxes, colors, scores)
        self._plot_boxes(frame_manip, boxes, colors, scores)

        self._show_fps(frame)

        cv2.imshow("Localizer", frame)
        cv2.imshow("Manip + NN", frame_manip)

        self._update_fps_counter()

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    
    def _plot_boxes(self, frame: np.ndarray, boxes: np.ndarray, colors, scores) -> None:
        color_black = (0, 0, 0)
        for i in range(boxes.shape[0]):
            box = boxes[i]
            y1 = (frame.shape[0] * box[0]).astype(int)
            y2 = (frame.shape[0] * box[2]).astype(int)
            x1 = (frame.shape[1] * box[1]).astype(int)
            x2 = (frame.shape[1] * box[3]).astype(int)
            color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
            cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)


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