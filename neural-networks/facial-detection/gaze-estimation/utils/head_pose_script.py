import depthai as dai
import numpy as np

try:
    while True:
        pose_msg = node.inputs["pose_input"].get()
        yaw = pose_msg.getTensor("tf.identity")
        pitch = pose_msg.getTensor("tf.identity_1")
        roll = pose_msg.getTensor("tf.identity_2")

        output = np.hstack([yaw, pitch, roll])
        output = np.array(output, dtype=np.float16)
        output_msg = dai.NNData()
        output_msg.addTensor("head_pose_angles_yaw_pitch_roll", output)
        output_msg.setTimestamp(pose_msg.getTimestamp())
        output_msg.setSequenceNum(pose_msg.getSequenceNum())

        node.outputs["head_pose_output"].send(output_msg)

except Exception as e:
    node.warn(str(e))
