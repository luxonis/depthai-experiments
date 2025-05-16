try:
    while True:
        frame = node.inputs["frame_input"].get()
        num_configs_message = node.inputs["num_configs_input"].get()
        conf_seq = num_configs_message.getSequenceNum()
        frame_seq = frame.getSequenceNum()
        num_configs = len(bytearray(num_configs_message.getData()))

        for i in range(num_configs):
            cfg = node.inputs["config_input"].get()
            node.outputs["output_config"].send(cfg)
            node.outputs["output_frame"].send(frame)

except Exception as e:
    node.warn(str(e))
