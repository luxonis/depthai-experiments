import depthai as dai
from typing import List

class DetectedKeypoints(dai.Buffer):
    def __init__(self, keypoints: List[dai.Buffer]):
        super().__init__()
        self.keypoints: List[dai.Buffer] = keypoints
        
class DetectionsKeypointsSync(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.keypoints_input = self.createInput(blocking=True)
        self.detections_input = self.createInput(blocking=True)
        self.passthrough_input = self.createInput(blocking=True)
        
        self.out = self.createOutput()
        
    def run(self) -> None:
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_message: dai.ImgDetections = self.detections_input.get()
            
            message_group = dai.MessageGroup()
            det_ts = detections_message.getTimestamp()
            
            keypoints = []
            for i, detection in enumerate(detections_message.detections):
                keypoints_output = self.keypoints_input.get()
                rec_ts = keypoints_output.getTimestamp()
                
                if rec_ts == det_ts:
                    keypoints.append(keypoints_output)
                else:
                    print("Recognition message is not in sync with detection message.")                    
                    break
                
            keypoints_message = DetectedKeypoints(keypoints)
            keypoints_message.setTimestamp(det_ts)
            message_group["detections"]= detections_message
            message_group["keypoints"]= keypoints_message
            message_group["passthrough"]= passthrough
                
            self.out.send(message_group)