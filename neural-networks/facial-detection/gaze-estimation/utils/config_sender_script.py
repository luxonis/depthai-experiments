try:
    while True:
        frame = node.inputs["frame_input"].get()
        configs_message = node.inputs["config_input"].get()

        conf_seq = configs_message.getSequenceNum()
        frame_seq = frame.getSequenceNum()

        messages = configs_message.getMessageNames()
        for msg_name in messages:
            cfg = configs_message[msg_name]
            node.outputs["output_config"].send(cfg)
            node.outputs["output_frame"].send(frame)

except Exception as e:
    node.warn(str(e))
