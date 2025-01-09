import depthai as dai
from typing import List

class Ages(dai.Buffer):
    def __init__(self, ages: List[dai.Buffer]):
        super().__init__()
        self.ages: List[dai.Buffer] = ages

class Genders(dai.Buffer):
    def __init__(self, genders: List[dai.Buffer]):
        super().__init__()
        self.genders: List[dai.Buffer] = genders
        
class DetectionsAgeGenderSync(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.age_input = self.createInput()
        self.gender_input = self.createInput()
        self.detections_input = self.createInput()
        self.passthrough_input = self.createInput()
        
        self.out = self.createOutput()
        self.out_frame = self.createOutput()
        
    def run(self) -> None:
        while self.isRunning():
            passthrough = self.passthrough_input.get()
            detections_message = self.detections_input.get()
            
            message_group = dai.MessageGroup()
            det_ts = detections_message.getTimestamp()
            
            ages = []
            genders = []
            
            for i, detection in enumerate(detections_message.detections):
                age_message = self.age_input.get()
                gender_message = self.gender_input.get()
                
                ages.append(age_message.predictions[0])
                genders.append(gender_message)

            ages = Ages(ages)
            ages.setTimestamp(det_ts)
            ages.setSequenceNum(passthrough.getSequenceNum())
            
            genders = Genders(genders)
            genders.setTimestamp(det_ts)
            genders.setSequenceNum(passthrough.getSequenceNum())
            
            message_group["detections"]= detections_message
            message_group["ages"]= ages
            message_group["genders"]= genders
            message_group.setTimestamp(det_ts)
            message_group.setSequenceNum(passthrough.getSequenceNum())
            
            self.out.send(message_group)
            self.out_frame.send(passthrough)
