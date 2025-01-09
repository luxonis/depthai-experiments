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
                print("getting recognition")
                recognition_output = self.recognitions_input.get()
                print("got recognition")
                if detections_message.getSequenceNum() != recognition_output.getSequenceNum():
                    print(f"[DetRectSync {recognition_output.getSequenceNum()}] Recognition message is not in sync with detection message. Det seq: {detections_message.getSequenceNum()}, Rec seq: {recognition_output.getSequenceNum()}")
                    
                recognitions.append(recognition_output)
                
            recognition_message = DetectedRecognitions(recognitions)
            recognition_message.setTimestamp(det_ts)
            recognition_message.setSequenceNum(passthrough.getSequenceNum())
            
            message_group["detections"]= detections_message
            message_group["recognitions"]= recognition_message
            message_group["passthrough"]= passthrough
            message_group.setTimestamp(det_ts)
            message_group.setSequenceNum(passthrough.getSequenceNum())
            print(f"[HostSync {recognition_message.getSequenceNum()}] Sending message group with {len(recognitions)} recognitions")
            self.out.send(message_group)