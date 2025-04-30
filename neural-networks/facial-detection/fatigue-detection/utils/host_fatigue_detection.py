import depthai as dai
from collections import deque
from utils.face_landmarks import determine_fatigue

from depthai_nodes.utils import AnnotationHelper


class FatigueDetection(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()

        self.crop_face = self.createInput()
        self.preview = self.createInput()
        self.face_nn = self.createInput()
        self.landmarks_nn = self.createInput()

        self.closed_eye_duration = deque(maxlen=30)
        self.head_tilted_duration = deque(maxlen=30)

        self.out = self.createOutput()
        self.output_frame = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            frame = self.preview.get().getCvFrame()
            face_dets = self.face_nn.get()
            dets = face_dets.detections
            n_detections = len(dets)

            annotations = AnnotationHelper()

            if n_detections >= 1:
                crop_face = self.crop_face.get().getCvFrame()
                landmarks = self.landmarks_nn.get()
                pitch, eyes_closed = determine_fatigue(frame, landmarks)

                for i in range(n_detections - 1):  # skip the rest of the detections
                    self.crop_face.get().getCvFrame()
                    self.landmarks_nn.get()

                self.head_tilted_duration.append(pitch)
                self.closed_eye_duration.append(eyes_closed)

                percent_closed_eyes = sum(self.closed_eye_duration) / len(
                    self.closed_eye_duration
                )
                percent_tilted = sum(self.head_tilted_duration) / len(
                    self.head_tilted_duration
                )

                print(percent_tilted, percent_closed_eyes)
                if percent_tilted >= 0.75:
                    print("tilted")
                    annotations.draw_text(
                        text="Head Tilted!",
                        position=(0.1, 0.1),
                    )

                if percent_closed_eyes >= 0.75:
                    print("closed")
                    annotations.draw_text(
                        text="Eyes Closed!",
                        position=(0.1, 0.2),
                    )

            annotations_msg = annotations.build(
                timestamp=face_dets.getTimestamp(),
                sequence_num=face_dets.getSequenceNum(),
            )

            self.out.send(annotations_msg)
