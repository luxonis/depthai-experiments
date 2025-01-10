import cv2
import depthai as dai
import numpy as np

from detected_recognitions import DetectedRecognitions


class PedestrianReidentification(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._results = []

    def build(
        self, img_frames: dai.Node.Output, detected_recognitions: dai.Node.Output
    ) -> "PedestrianReidentification":
        self.link_args(img_frames, detected_recognitions)
        self.sendProcessingToPipeline(True)
        return self

    def process(
        self, img_frame: dai.ImgFrame, detected_recognitions: dai.Buffer
    ) -> None:
        frame = img_frame.getCvFrame()
        assert isinstance(detected_recognitions, DetectedRecognitions)
        detections = detected_recognitions.img_detections.detections

        for i, detection in enumerate(detections):
            bbox = self._frame_norm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )

            reid_result = detected_recognitions.nn_data[i].getFirstTensor().flatten()

            for i, vector in enumerate(self._results):
                dist = self._cos_dist(reid_result, vector)
                if dist > 0.7:
                    self._results[i] = np.array(reid_result)
                    break
            else:
                self._results.append(np.array(reid_result))

            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2
            )
            y = (bbox[1] + bbox[3]) // 2
            cv2.putText(
                frame,
                f"Person {i}",
                (bbox[0], y),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.5,
                (0, 0, 0),
                8,
            )
            cv2.putText(
                frame,
                f"Person {i}",
                (bbox[0], y),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.5,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()

    def _cos_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _frame_norm(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
