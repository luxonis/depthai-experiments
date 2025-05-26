import depthai as dai

class SyncNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.inputPCL = self.createInput()
        self.inputDet= self.createInput()
        self.inputRGB= self.createInput()

        self.out = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            pcl_msg = self.inputPCL.get()
            detections_msg = self.inputDet.get()
            rgb_msg = self.inputRGB.get()
            timestamp = detections_msg.getTimestamp()

            # print(f"SyncNode: Received PCL: {pcl_msg.getSequenceNum()}, Detections: {detections_msg.getSequenceNum()}, RGB: {rgb_msg.getSequenceNum()}")
            # print(pcl_msg.getSequenceNum() == detections_msg.getSequenceNum() == rgb_msg.getSequenceNum())

            message_group = dai.MessageGroup()
            message_group["pcl"] = pcl_msg
            message_group["detections"] =detections_msg 
            message_group["rgb"] = rgb_msg
            message_group.setTimestamp(timestamp)
            message_group.setSequenceNum(detections_msg.getSequenceNum())

            self.out.send(message_group)
