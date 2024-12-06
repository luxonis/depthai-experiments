from queue import PriorityQueue
import time
import depthai as dai
from detected_recognitions import DetectedRecognitions


class DetectionsRecognitionsSync(dai.node.ThreadedHostNode):
    FPS_TOLERANCE_DIVISOR = 2.0
    INPUT_CHECKS_PER_FPS = 100

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._camera_fps = 30
        self._unmatched_recognitions: list[dai.NNData] = []
        self._recognitions_by_detection_ts: dict[float, list[dai.NNData]] = {}
        self._detections: dict[float, dai.ImgDetections | dai.SpatialImgDetections] = {}
        self._ready_timestamps = PriorityQueue()

        self.input_recognitions = dai.Node.Input(self)
        self.input_detections = dai.Node.Input(self)
        self.output = dai.Node.Output(self)
        
    
    def build(self) -> "DetectionsRecognitionsSync":
        return self
    

    def set_camera_fps(self, fps: int) -> None:
        self._camera_fps = fps


    def run(self) -> None:
        while self.isRunning():
            input_recognitions = self.input_recognitions.tryGet()
            if input_recognitions:
                self._add_recognition(input_recognitions)

            input_detections = self.input_detections.tryGet()
            if input_detections:
                self._add_detection(input_detections)

            if input_recognitions or input_detections:
                ready_data = self._pop_ready_data()
                if ready_data:
                    self._clear_old_data(ready_data)
                    self.output.send(ready_data)
            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)


    def _add_recognition(self, recognition: dai.NNData) -> None:
        recognition_ts = self._get_total_seconds_ts(recognition)
        best_matching_detection_ts = self._get_matching_detection_ts(recognition_ts)
        
        if best_matching_detection_ts is not None:
            self._add_recognition_by_detection_ts(recognition, best_matching_detection_ts)
            self._update_ready_timestamps(best_matching_detection_ts)
        else:
            self._unmatched_recognitions.append(recognition)


    def _get_matching_detection_ts(self, recognition_ts: float) -> float | None:
        for detection_ts in self._detections.keys():
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                return detection_ts
        return None
    

    def _add_detection(self, detection: dai.ImgDetections | dai.SpatialImgDetections) -> None:
        detection_ts = self._get_total_seconds_ts(detection)
        self._detections[detection_ts] = detection
        self._try_match_recognitions(detection_ts)
        self._update_ready_timestamps(detection_ts)


    def _try_match_recognitions(self, detection_ts: float) -> None:
        matched_recognitions: list[dai.NNData] = []
        for recognition in self._unmatched_recognitions:
            recognition_ts = self._get_total_seconds_ts(recognition)
            if self._timestamps_in_tolerance(detection_ts, recognition_ts):
                self._add_recognition_by_detection_ts(recognition, detection_ts)
                matched_recognitions.append(recognition)
        
        for matched_recognition in matched_recognitions:
            self._unmatched_recognitions.remove(matched_recognition)

    
    def _timestamps_in_tolerance(self, timestamp1: float, timestamp2: float) -> bool:
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / self.FPS_TOLERANCE_DIVISOR)


    def _add_recognition_by_detection_ts(self, recognition: dai.NNData, detection_ts: float) -> None:
        if detection_ts in self._recognitions_by_detection_ts:
            self._recognitions_by_detection_ts[detection_ts].append(recognition)
        else:
            self._recognitions_by_detection_ts[detection_ts] = [recognition]


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
        
        recognitions = self._recognitions_by_detection_ts.get(timestamp)
        if not recognitions:
            return False
        
        return len(detections.detections) == len(recognitions)


    def _pop_ready_data(self) -> DetectedRecognitions | None:
        if self._ready_timestamps.empty():
            return None
        
        timestamp = self._ready_timestamps.get()
        detections = self._detections.pop(timestamp)
        recognitions = self._recognitions_by_detection_ts.pop(timestamp, None)

        return DetectedRecognitions(detections, recognitions)

    
    def _clear_old_data(self, ready_data: DetectedRecognitions) -> None:
        current_timestamp = self._get_total_seconds_ts(ready_data)
        self._clear_unmatched_recognitions(current_timestamp)
        self._clear_old_detections(current_timestamp)


    def _clear_unmatched_recognitions(self, current_timestamp) -> None:
        unmatched_recognitions_to_remove = []
        for unmatched_recognition in self._unmatched_recognitions:
            if self._get_total_seconds_ts(unmatched_recognition) < current_timestamp:
                unmatched_recognitions_to_remove.append(unmatched_recognition)
        
        for unmatched_recognition in unmatched_recognitions_to_remove:
            self._unmatched_recognitions.remove(unmatched_recognition)


    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

        
    def _clear_old_detections(self, current_timestamp) -> None:
        detection_keys_to_pop = []
        for detection_ts in self._detections.keys():
            if detection_ts < current_timestamp:
                detection_keys_to_pop.append(detection_ts)
        
        for detection_ts in detection_keys_to_pop:
            self._detections.pop(detection_ts)
            self._recognitions_by_detection_ts.pop(detection_ts, None)
