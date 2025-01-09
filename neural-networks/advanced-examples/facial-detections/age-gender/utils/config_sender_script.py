try:
    while True:
        frame = node.inputs['frame_input'].get()
        node.warn(f"[ConfigSender {frame.getSequenceNum()}] Got frame {frame.getTimestamp()}")
        
        configs_message = node.inputs['config_input'].get()
        node.warn(f"[ConfigSender {frame.getSequenceNum()}] Got configs ")       
        conf_seq = configs_message.getSequenceNum()
        frame_seq = frame.getSequenceNum()
        
        while conf_seq > frame_seq:
            node.warn(f"[ConfigSender {conf_seq}] Configs {conf_seq} mismatch with frame {frame_seq}")
            frame = node.inputs['frame_input'].get()
            
        messages = configs_message.getMessageNames()
        
        node.warn(f"[ConfigSender {conf_seq}] got {len(messages)} messages with ts {configs_message.getTimestamp()} and frame seq {frame.getSequenceNum()}")
        for i, cfg in configs_message:
            node.warn(f"[ConfigSender {conf_seq}] sending {i}")
            node.outputs['output_config'].send(cfg)
            node.outputs['output_frame'].send(frame)
            node.warn(f"[ConfigSender {conf_seq}] sent {i}")
            
except Exception as e:
    node.warn(str(e))