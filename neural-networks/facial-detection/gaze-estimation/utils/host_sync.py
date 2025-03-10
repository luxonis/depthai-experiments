import depthai as dai
from typing import List


class Landmarks(dai.Buffer):
    def __init__(self, landmarks: List[dai.Buffer]):
        super().__init__()
        self.landmarks: List[dai.Buffer] = landmarks


class ImageLandmarkSync(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.frame_input = self.createInput()
        self.detections_input = self.createInput()
        self.gaze_input = self.createInput()

        self.out = self.createOutput()
        self.frame_out = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            frame = self.frame_input.get()
            detections_message = self.detections_input.get()

            timestamp = detections_message.getTimestamp()
            sequenece_num = detections_message.getSequenceNum()
            message_group = dai.MessageGroup()
            gaze_messages = []
            for _ in detections_message.detections:
                gaze_message = self.gaze_input.get()
                gaze_messages.append(gaze_message)

            gazes = Landmarks(gaze_messages)
            gazes.setTimestamp(timestamp)
            gazes.setSequenceNum(sequenece_num)

            message_group["detections"] = detections_message
            message_group["gazes"] = gazes
            message_group.setTimestamp(timestamp)
            message_group.setSequenceNum(sequenece_num)

            self.out.send(message_group)
            self.frame_out.send(frame)
