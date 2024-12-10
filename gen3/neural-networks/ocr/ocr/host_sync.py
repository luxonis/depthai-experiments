import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import ImgDetectionsExtended

 
class DetectedRecognitions(dai.Buffer):
    def __init__(self, det: ImgDetectionsExtended, classes: list[str]):
        super().__init__()
        self.detections: ImgDetectionsExtended = det.detections
        self.classes: list[str] = classes
        
        self.setTimestamp(det.getTimestamp())

class CustomSyncNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.ocr_inputs = self.createInput(blocking=False)
        self.detections_inputs = self.createInput(blocking=False)
        self.passthrough_input = self.createInput(blocking=False)
        
        self.output = self.createOutput()
        self.passthrough = self.createOutput()
        
    def run(self) -> None:
        
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_queue = self.detections_inputs.get()
            
            
            ocrs = []
            # print(f"Processing {len(detections_queue.detections)} ocrs.")
            for i, detection in enumerate(detections_queue.detections):
                # print(f"waiting for ocr recognition to finish for detection: {i}")
                ocr_output = self.ocr_inputs.get()
                
                ocrs.append( ocr_output.classes)
                
            
            det_recog = DetectedRecognitions(detections_queue, ocrs)
            
            self.output.send(det_recog)
            self.passthrough.send(passthrough)
        