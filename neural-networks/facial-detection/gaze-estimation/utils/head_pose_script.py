import depthai as dai
import numpy as np

try:
    while True:
        pose_msg = node.inputs["pose_input"].get()

        yaw = pose_msg.getTensor("tf.identity")
        pitch = pose_msg.getTensor("tf.identity_1")
        roll = pose_msg.getTensor("tf.identity_2")

        # yaw = yaw_msg.predictions[0].prediction
        # pitch = pitch_msg.predictions[0].prediction
        # roll = roll_msg.predictions[0].prediction

        output = np.hstack([yaw, pitch, roll])
        output = np.array(output, dtype=np.float16)
        # output = output.tolist()
        output_msg = dai.NNData()
        output_msg.addTensor("head_pose_angles_yaw_pitch_roll", output)

        # node.warn(f"type: {type(output_msg)}, data: {output_msg.getData()}, layers: {output_msg.getAllLayerNames()}")

        node.outputs["head_pose_output"].send(output_msg)
        # node.warn("Sent head pose")


except Exception as e:
    node.warn(str(e))
