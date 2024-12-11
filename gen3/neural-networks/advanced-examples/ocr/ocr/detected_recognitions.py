import depthai as dai

class DetectedRecognitions(dai.Buffer):
    def __init__(self, detections: dai.ImgDetections, data: list[dai.Buffer]) -> None:
        super().__init__(0)
        self.img_detections: dai.ImgDetections = detections
        self.data: list[dai.Buffer] = data
        
        self.setTimestampDevice(detections.getTimestampDevice())
        self.setTimestamp(detections.getTimestamp())
        self.setSequenceNum(detections.getSequenceNum())