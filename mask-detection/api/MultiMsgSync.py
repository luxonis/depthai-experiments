# Color frames (ImgFrame), object detection (ImgDetections) and recognition (NNData)
# messages arrive to the host all with some additional delay.
# For each ImgFrame there's one ImgDetections msg, which has multiple detections, and for each
# detection there's a NNData msg which contains recognition results.

# How it works:
# Every ImgFrame, ImgDetections and NNData message has it's own sequence number, by which we can sync messages.

class TwoStageHostSeqSync:
    def __init__(self):
        self.msgs = {}
    # name: color, detection, or recognition
    def add_msg(self, msg, name):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {} # Create directory for msgs
        if "recognition" not in self.msgs[seq]:
            self.msgs[seq]["recognition"] = [] # Create recognition array

        if name == "recognition":
            # Append recognition msgs to an array
            self.msgs[seq]["recognition"].append(msg)
            # print(f'Added recognition seq {seq}, total len {len(self.msgs[seq]["recognition"])}')

        elif name == "detection":
            # Save detection msg in the directory
            self.msgs[seq][name] = msg
            self.msgs[seq]["len"] = len(msg.detections)
            # print(f'Added detection seq {seq}')

        elif name == "color": # color
            # Save color frame in the directory
            self.msgs[seq][name] = msg
            # print(f'Added frame seq {seq}')


    def get_msgs(self):
        seq_remove = [] # Arr of sequence numbers to get deleted

        for seq, msgs in self.msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair

            # Check if we have both detections and color frame with this sequence number
            if "color" in msgs and "len" in msgs:

                # Check if all detected objects (faces) have finished recognition inference
                if msgs["len"] == len(msgs["recognition"]):
                    # print(f"Synced msgs with sequence number {seq}", msgs)

                    # We have synced msgs, remove previous msgs (memory cleaning)
                    for rm in seq_remove:
                        del self.msgs[rm]

                    return msgs # Returned synced msgs

        return None # No synced msgs