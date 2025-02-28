import depthai as dai

try:
    while True:
        yaw_msg = node.inputs["yaw_input"].get()
        pitch_msg = node.inputs["pitch_input"].get()
        roll_msg = node.inputs["roll_input"].get()

        yaw = yaw_msg.predictions[0].prediction
        pitch = pitch_msg.predictions[0].prediction
        roll = roll_msg.predictions[0].prediction

        output = [[yaw, pitch, roll]]
        output_msg = dai.NNData()
        output_msg.setData(output)

        node.outputs["head_pose_output"].send(output_msg)


except Exception as e:
    node.warn(str(e))
