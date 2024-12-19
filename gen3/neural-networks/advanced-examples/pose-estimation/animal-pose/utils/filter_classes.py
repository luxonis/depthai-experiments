import depthai as dai

class FilterClasses(dai.node.ThreadedHostNode):
    def __init__(self, labels: list[int]) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.out = self.createOutput()
        self.labels = labels

    def run(self):
        if not self.labels:
            raise ValueError("No labels provided.")
        while self.isRunning():
            try:
                incoming_detections: dai.ImgDetections = self.input_detections.get()
            except dai.MessageQueue.QueueException:
                break

            filtered_detections = []

            for detection in incoming_detections.detections:
                if detection.label in self.labels:
                    filtered_detections.append(detection)

            incoming_detections.detections = filtered_detections
            self.out.send(incoming_detections)