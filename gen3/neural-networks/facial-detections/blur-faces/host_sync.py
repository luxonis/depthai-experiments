class HostSync:
    def __init__(self):
        self.arrays = {}


    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append(msg)


    def get_msgs(self, seq):
        ret = {}
        for name, arr in self.arrays.items():
            for i, msg in enumerate(arr):
                if msg.getSequenceNum() == seq:
                    ret[name] = msg
                    self.arrays[name] = arr[i:]
                    break
        return ret
