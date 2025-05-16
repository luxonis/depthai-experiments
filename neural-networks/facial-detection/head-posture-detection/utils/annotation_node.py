import depthai as dai
import numpy as np
from depthai_nodes.utils import AnnotationHelper

SECONDARY_COLOR = dai.Color(
    float(240 / 255), float(240 / 255), float(240 / 255), float(1.0)
)


class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()

        self.output_annotation = self.createOutput()
        self.output_frame = self.createOutput()

        self._min_threshold = 15

    def run(self):
        while self.isRunning():
            synced_data = self.input.get()
            annotation_helper = AnnotationHelper()

            frame = synced_data.reference_data  # dai.ImgFrame

            detections_pose = synced_data.gathered

            detections = detections_pose[0].reference_data.detections
            poses = detections_pose[0].gathered

            for det, pose in zip(detections, poses):
                yaw = pose.getTensor("tf.identity").ravel()[0]
                roll = pose.getTensor("tf.identity_1").ravel()[0]
                pitch = pose.getTensor("tf.identity_2").ravel()[0]

                pose_text = self._decode_pose(yaw, pitch, roll)

                pose_information = (
                    f"Pitch: {pitch:.0f} \nYaw: {yaw:.0f} \nRoll: {roll:.0f}"
                )

                outer_points = det.rotated_rect.getOuterRect()

                x_min, y_min, x_max, y_max = [np.round(x, 2) for x in outer_points]

                annotation_helper.draw_rectangle(
                    (x_min, y_min),
                    (x_max, y_max),
                    fill_color=dai.Color(0.0, 0.0, 0.0, 0.0),
                )

                annotation_helper.draw_text(
                    pose_information,
                    (x_max, y_min + 0.1),
                    size=16,
                    color=SECONDARY_COLOR,
                )
                annotation_helper.draw_text(pose_text, (x_min, y_min), size=28)

            annotations = annotation_helper.build(
                timestamp=frame.getTimestamp(),
                sequence_num=frame.getSequenceNum(),
            )

            self.output_frame.send(frame)
            self.output_annotation.send(annotations)

    def _decode_pose(self, yaw: float, pitch: float, roll: float) -> str:
        vals = np.array([abs(pitch), abs(yaw), abs(roll)])
        max_index = np.argmax(vals)

        if vals[max_index] < self._min_threshold:
            return ""

        if max_index == 0:
            if pitch > 0:
                txt = "Look down"
            else:
                txt = "Look up"
        elif max_index == 1:
            if yaw > 0:
                txt = "Turn left"
            else:
                txt = "Turn right"
        elif max_index == 2:
            if roll > 0:
                txt = "Tilt left"
            else:
                txt = "Tilt right"

        return txt
