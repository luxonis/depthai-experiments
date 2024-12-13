import depthai as dai
from typing import List

class DetectedRecognitions(dai.Buffer):
    def __init__(self, classes: List[str]):
        super().__init__()
        self.classes: List[str] = classes
        
class CustomSyncNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.ocr_inputs = self.createInput(blocking=True)
        self.detections_inputs = self.createInput(blocking=True)
        self.passthrough_input = self.createInput(blocking=True)
        
        self.out = self.createOutput()
        
    def run(self) -> None:
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_queue = self.detections_inputs.get()
            
            message_group = dai.MessageGroup()
            det_ts = detections_queue.getTimestamp()
            
            ocrs = []
            ocr_ts = det_ts
            
            print(f"Sync num detections: {len(detections_queue.detections)}")
            for i, detection in enumerate(detections_queue.detections):
                ocr_output = self.ocr_inputs.get()
                
                ocr_ts = ocr_output.getTimestamp()
                if ocr_ts != det_ts:
                    print(f"ocr ts: {ocr_ts}, det ts: {det_ts}")
                    print(f"Timestamp mismatch!!")
                    
                text = "".join(ocr_output.classes)
                
                ocrs.append(text)
                
            ocr_message = DetectedRecognitions(ocrs)
            ocr_message.setTimestamp(det_ts)
            message_group["detections"]= detections_queue
            message_group["ocrs"]= ocr_message
            message_group["passthrough"]= passthrough
                
            self.out.send(message_group)
        