from queue import PriorityQueue
import depthai as dai
from detected_recognitions import DetectedRecognitions


class DetectionsRecognitionsSync(dai.node.ThreadedHostNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._recognitions: dict[int, list[dai.NNData]] = {}
        self._detections: dict[int, dai.ImgDetections] = {}
        self._ready_timestamps = PriorityQueue()

        self.input_recognitions = dai.Node.Input(self)
        self.input_detections = dai.Node.Input(self)
        self.output = dai.Node.Output(self)
        
    
    def build(self) -> "DetectionsRecognitionsSync":
        return self


    def run(self) -> None:
        while self.isRunning():
            detection: dai.ImgDetections = self.input_detections.get()
            detection_ts = detection.getTimestamp().total_seconds()
            self._detections[detection_ts] = detection
            self._update_ready_timestamps(detection_ts)
            
            if self.input_recognitions.has():
                recognition: dai.NNData = self.input_recognitions.get()
                recognition_ts = recognition.getTimestamp().total_seconds()
                if recognition.getTimestamp() not in self._recognitions:
                    self._recognitions[recognition_ts] = [recognition]
                else:
                    self._recognitions[recognition_ts].append(recognition)
                self._update_ready_timestamps(recognition_ts)

            ready_data = self._get_ready_data()
            if ready_data:
                self.output.send(ready_data)


    def _update_ready_timestamps(self, timestamp: float) -> None:
        if not self._timestamp_ready(timestamp):
            return
        
        self._ready_timestamps.put(timestamp)
        

    def _timestamp_ready(self, timestamp: float) -> None:
        detections = self._detections.get(timestamp)
        if not detections:
            return False
        elif len(detections.detections) == 0:
            return True
        
        recognitions = self._recognitions.get(timestamp)
        if not recognitions:
            return False
        
        return len(detections.detections) == len(recognitions)


    def _get_ready_data(self) -> DetectedRecognitions | None:
        if self._ready_timestamps.empty():
            return None
        
        timestamp = self._ready_timestamps.get()
        detections = self._detections.get(timestamp)
        recognitions = self._recognitions.get(timestamp)

        self._clear_timestamp(timestamp)

        return DetectedRecognitions(detections, recognitions)


    def _clear_timestamp(self, timestamp: float) -> None:
        self._recognitions.pop(timestamp, None)
        self._detections.pop(timestamp)