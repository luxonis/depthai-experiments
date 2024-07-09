import depthai as dai


class DetectedRecognitions(dai.Buffer):
    def __init__(self, detections: dai.ImgDetections, nn_data: list[dai.NNData]) -> None:
        super().__init__(0)
        self.detections: dai.ImgDetections = detections
        self.nn_data: list[dai.NNData] = nn_data
        
        self.setTimestampDevice(detections.getTimestampDevice())
        self.setTimestamp(detections.getTimestamp()) # Can be removed
        self.setSequenceNum(detections.getSequenceNum()) # Can be removed