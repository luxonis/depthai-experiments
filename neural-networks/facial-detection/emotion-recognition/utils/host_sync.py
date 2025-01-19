import depthai as dai
from depthai_nodes.ml.messages import Classifications
import numpy as np


class DetectionSyncNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.emotion_input = self.createInput()
        self.detections_input = self.createInput()
        self.passthrough_input = self.createInput()

        self.out = self.createOutput()
        self.out_frame = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_message = self.detections_input.get()
            det_ts = detections_message.getTimestamp()

            emotions = []
            emotions_prob = []
            for i, detection in enumerate(detections_message.detections):
                emotion_message = self.emotion_input.get()

                emotions.append(emotion_message.top_class)
                emotions_prob.append(emotion_message.top_score)

            emotion_message = Classifications()
            emotion_message.classes = emotions
            emotion_message.scores = np.array(emotions_prob)
            emotion_message.setTimestamp(det_ts)
            emotion_message.setSequenceNum(passthrough.getSequenceNum())

            message_group = dai.MessageGroup()
            message_group["detections"] = detections_message
            message_group["emotions"] = emotion_message
            message_group.setTimestamp(det_ts)
            message_group.setSequenceNum(passthrough.getSequenceNum())

            self.out.send(message_group)
            self.out_frame.send(passthrough)
