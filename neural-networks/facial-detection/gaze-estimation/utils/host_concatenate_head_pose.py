import depthai as dai
import numpy as np


class ConcatenateHeadPose(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self.pose_input = self.createInput()
        self.head_pose_output = self.createOutput()

    def build(
        self,
        pose_intput: dai.Node.Output,
    ) -> "ConcatenateHeadPose":
        self.link_args(pose_intput)
        return self

    def process(self, pose_intput: dai.NNData) -> None:
        timestamp = pose_intput.getTimestamp()
        sequence_num = pose_intput.getSequenceNum()

        yaw = pose_intput.getTensor("tf.identity")
        pitch = pose_intput.getTensor("tf.identity_1")
        roll = pose_intput.getTensor("tf.identity_2")

        output_list = np.hstack([yaw, pitch, roll])
        output_message = dai.NNData()
        output_message.addTensor(
            "head_pose_angles_yaw_pitch_roll", output_list, dai.TensorInfo.DataType.FP16
        )
        output_message.setTimestamp(timestamp)
        output_message.setSequenceNum(sequence_num)

        self.head_pose_output.send(output_message)
