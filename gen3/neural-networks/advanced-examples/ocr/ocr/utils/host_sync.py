import depthai as dai
from typing import List

class DetectedRecognitions(dai.Buffer):
    def __init__(self, recognitions: List[dai.Buffer]):
        super().__init__()
        self.recognitions: List[dai.Buffer] = recognitions
        
class DetectionsRecognitionsSync(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.recognitions_input = self.createInput(blocking=True)
        self.detections_input = self.createInput(blocking=True)
        self.passthrough_input = self.createInput(blocking=True)
        
        self.out = self.createOutput()
        
    def run(self) -> None:
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_message = self.detections_input.get()
            
            message_group = dai.MessageGroup()
            det_ts = detections_message.getTimestamp()
            
            recognitions = []
            for i, detection in enumerate(detections_message.detections):
                recognition_output = self.recognitions_input.get()
                rec_ts = recognition_output.getTimestamp()
                
                if rec_ts == det_ts:
                    recognitions.append(recognition_output)
                else:
                    print("Recognition message is not in sync with detection message.")                    
                    break
                
            recognition_message = DetectedRecognitions(recognitions)
            recognition_message.setTimestamp(det_ts)
            message_group["detections"]= detections_message
            message_group["recognitions"]= recognition_message
            message_group["passthrough"]= passthrough
                
            self.out.send(message_group)
