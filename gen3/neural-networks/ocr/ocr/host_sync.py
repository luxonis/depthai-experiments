import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import ImgDetectionExtended

 
class DetectedRecognition(dai.Buffer):
    def __init__(self, detection: ImgDetectionExtended, classes: list[str]):
        super().__init__()
        self.detection = detection
        self.classes = classes
        
        self.setTimestamp(detection.getTimestamp())
        
class SyncedData(dai.Buffer):
    def __init__(self, detections: list[DetectedRecognition], passthrough: dai.ImgFrame):
        super().__init__()
        self.detections = detections
        self.passthrough = passthrough
        

class CustomSyncNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.ocr_inputs = self.createInput(blocking=False)
        self.detections_inputs = self.createInput(blocking=False)
        self.passthrough_input = self.createInput(blocking=False)
        
        self.output = self.createOutput()
        
    def run(self) -> None:
        
        while self.isRunning():
            detections = self.detections_inputs.get().detections
            passthrough = self.passthrough_input.get()
            ocrs = []
            print(f"Processing {len(detections)} ocrs.")
            for i, detection in enumerate(detections):
                ocr_output = self.ocr_inputs.get()
                print(f"waiting for ocr recognition to finish for detection: {i}")
                print("ocr output received")
                print(ocr_output.classes)
                ocrs.append(DetectedRecognition(detection, ocr_output.classes))
                break
            
            print("sending synced data")
            self.output.send(SyncedData(ocrs, passthrough))
        