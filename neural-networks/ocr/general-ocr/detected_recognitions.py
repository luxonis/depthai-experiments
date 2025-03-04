import depthai as dai
from typing import List


class DetectedRecognitions(dai.Buffer):
    def __init__(self, detections: dai.ImgDetections, data: List[dai.Buffer]) -> None:
        super().__init__(0)
        self.img_detections: dai.ImgDetections = detections
        self.data: List[dai.Buffer] = data

        self.setTimestampDevice(detections.getTimestampDevice())
        self.setTimestamp(detections.getTimestamp())
        self.setSequenceNum(detections.getSequenceNum())
