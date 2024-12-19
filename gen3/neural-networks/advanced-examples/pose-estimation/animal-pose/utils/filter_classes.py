import depthai as dai

class FilterClasses(dai.node.ThreadedHostNode):
    def __init__(self, labels: list[int], only_one_detection: bool) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.out = self.createOutput()
        self.labels = labels
        self.only_one_detection = only_one_detection

    def run(self):
        if not self.labels:
            raise ValueError("No labels provided.")
        while self.isRunning():
            try:
                incoming_detections: dai.ImgDetections = self.input_detections.get()
            except dai.MessageQueue.QueueException:
                break

            filtered_detections = []

            best_detection = None
            best_score = 0

            for detection in incoming_detections.detections:
                if detection.label in self.labels:
                    if detection.confidence > best_score:
                        best_detection = detection
                        best_score = detection.confidence
                    if not self.only_one_detection:
                        filtered_detections.append(detection)

            if self.only_one_detection and best_detection is not None:
                filtered_detections.append(best_detection)

            incoming_detections.detections = filtered_detections
            self.out.send(incoming_detections)