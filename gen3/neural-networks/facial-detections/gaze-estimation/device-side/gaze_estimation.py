import cv2
import depthai as dai
import numpy as np
from detected_recognitions import DetectedRecognitions


class GazeEstimation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self,
              img_frames: dai.Node.Output,
              gaze_synced: dai.Node.Output,
              landmarks_synced: dai.Node.Output
        ) -> "GazeEstimation":
        self.sendProcessingToPipeline(True)
        self.link_args(img_frames, gaze_synced, landmarks_synced)
        return self
    

    def process(self, 
                img_frame: dai.ImgFrame, 
                gaze_synced: dai.Buffer,
                landmarks_synced: dai.Buffer
        ) -> None:
        assert(isinstance(gaze_synced, DetectedRecognitions))
        assert(isinstance(landmarks_synced, DetectedRecognitions))
        
        frame: np.ndarray = img_frame.getCvFrame()
        dets: list[dai.ImgDetection] = gaze_synced.img_detections.detections
        gaze_nn = gaze_synced.nn_data
        landmarks_nn = landmarks_synced.nn_data
        for i, detection in enumerate(dets):
            tl, br = self._denormalize_bounding_box(detection, frame.shape)
            cv2.rectangle(frame, tl, br, (10, 245, 10), 1)
            gaze_first_layer = gaze_nn[i].getAllLayerNames()[0]
            gaze = gaze_nn[i].getTensor(gaze_first_layer).astype(np.float16).flatten()
            gaze_x, gaze_y = (gaze * 100).astype(int)[:2]

            landmarks_first_layer = landmarks_nn[i].getAllLayerNames()[0]
            landmarks = landmarks_nn[i].getTensor(landmarks_first_layer).astype(np.float16).flatten()
            colors = [(0, 127, 255), (0, 127, 255), (255, 0, 127), (127, 255, 0), (127, 255, 0)]
            for lm_i in range(0, len(landmarks) // 2):
                x, y = landmarks[lm_i*2:lm_i*2+2]
                point = self._map_denormalized_point(x, y, detection, frame.shape)

                if lm_i <= 1: # Draw arrows from left eye & right eye
                    cv2.arrowedLine(frame, point, ((point[0] + gaze_x*5), (point[1] - gaze_y*5)), colors[lm_i], 3)

                cv2.circle(frame, point, 2, colors[lm_i], 2)

        cv2.imshow("Lasers", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _denormalize_bounding_box(self, detection: dai.ImgDetection, frame_shape: tuple) -> tuple:
        return (
                (int(frame_shape[1] * detection.xmin), int(frame_shape[0] * detection.ymin)),
                (int(frame_shape[1] * detection.xmax), int(frame_shape[0] * detection.ymax))
            )


    def _map_denormalized_point(self, x: int, y: int, detection: dai.ImgDetection, frame_shape: tuple) -> tuple[int, int]:
        width = detection.xmax - detection.xmin
        mapped_x = detection.xmin + width * x

        height = detection.ymax - detection.ymin
        mapped_y = detection.ymin + height * y

        return int(mapped_x * frame_shape[1]), int(mapped_y * frame_shape[0])