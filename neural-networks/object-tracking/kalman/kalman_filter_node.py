import numpy as np
import cv2
import depthai as dai
from typing import List

from kalman_filter import KalmanFilter


class KalmanFilterNode(dai.node.HostNode):
    def __init__(self):
        self._kalman_filters = {}
        super().__init__()

    def build(
        self,
        rgb: dai.Node.Output,
        tracker_out: dai.Node.Output,
        baseline: float,
        focal_length: float,
        label_map: List[str],
    ) -> "KalmanFilterNode":
        self.link_args(rgb, tracker_out)
        self.sendProcessingToPipeline(True)
        self._baseline = baseline
        self._focal_length = focal_length
        self._label_map = label_map
        return self

    def process(self, img_frame: dai.ImgFrame, tracklets: dai.Tracklets) -> None:
        frame: np.ndarray = img_frame.getCvFrame()
        current_time = tracklets.getTimestamp()

        for t in tracklets.tracklets:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            x_space = t.spatialCoordinates.x
            y_space = t.spatialCoordinates.y
            z_space = t.spatialCoordinates.z

            meas_vec_bbox = np.array(
                [[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]]
            )
            meas_vec_space = np.array([[x_space], [y_space], [z_space]])
            meas_std_space = z_space**2 / (self._baseline * self._focal_length)

            if t.status.name == "NEW":
                # Adjust these parameters
                acc_std_space = 10
                acc_std_bbox = 0.1
                meas_std_bbox = 0.05

                self._kalman_filters[t.id] = {
                    "bbox": KalmanFilter(
                        meas_std_bbox, acc_std_bbox, meas_vec_bbox, current_time
                    ),
                    "space": KalmanFilter(
                        meas_std_space, acc_std_space, meas_vec_space, current_time
                    ),
                }

            else:
                dt = current_time - self._kalman_filters[t.id]["bbox"].time
                dt = dt.total_seconds()
                self._kalman_filters[t.id]["space"].meas_std = meas_std_space

                if t.status.name != "TRACKED":
                    meas_vec_bbox = None
                    meas_vec_space = None

                if z_space == 0:
                    meas_vec_space = None

                self._kalman_filters[t.id]["bbox"].predict(dt)
                self._kalman_filters[t.id]["bbox"].update(meas_vec_bbox)

                self._kalman_filters[t.id]["space"].predict(dt)
                self._kalman_filters[t.id]["space"].update(meas_vec_space)

                self._kalman_filters[t.id]["bbox"].time = current_time
                self._kalman_filters[t.id]["space"].time = current_time

                vec_bbox = self._kalman_filters[t.id]["bbox"].x
                vec_space = self._kalman_filters[t.id]["space"].x

                x1_filter = int(vec_bbox[0] - vec_bbox[2] / 2)
                x2_filter = int(vec_bbox[0] + vec_bbox[2] / 2)
                y1_filter = int(vec_bbox[1] - vec_bbox[3] / 2)
                y2_filter = int(vec_bbox[1] + vec_bbox[3] / 2)

                cv2.rectangle(
                    frame,
                    (x1_filter, y1_filter),
                    (x2_filter, y2_filter),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"X: {int(vec_space[0])} mm",
                    (x1 + 10, y1 + 110),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 255),
                )
                cv2.putText(
                    frame,
                    f"Y: {int(vec_space[1])} mm",
                    (x1 + 10, y1 + 125),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 255),
                )
                cv2.putText(
                    frame,
                    f"Z: {int(vec_space[2])} mm",
                    (x1 + 10, y1 + 140),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 255),
                )

            try:
                label = self._label_map[t.label]
            except Exception:
                label = t.label

            cv2.putText(
                frame,
                str(label),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"ID: {[t.id]}",
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                t.status.name,
                (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))

            cv2.putText(
                frame,
                f"X: {int(x_space)} mm",
                (x1 + 10, y1 + 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"Y: {int(y_space)} mm",
                (x1 + 10, y1 + 80),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"Z: {int(z_space)} mm",
                (x1 + 10, y1 + 95),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )

        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()
