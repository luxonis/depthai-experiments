import numpy as np
from functools import reduce
from collections import deque
from typing import List

class HostSync:
    def __init__(self, streams: List[str], maxlen=50):
        self.queues = {stream: deque(maxlen=maxlen) for stream in streams}

    def add(self, stream: str, msg):
        self.queues[stream].append({'msg': msg, 'seq': msg.getSequenceNum()})

    def get(self):
        seqs = [np.array([msg['seq'] for msg in msgs]) for msgs in self.queues.values()]
        matching_seqs = reduce(np.intersect1d, seqs)
        if len(matching_seqs) == 0:
            return None
        seq = np.max(matching_seqs)
        res = {stream: next(msg['msg'] for msg in msgs if msg['seq'] == seq) for stream, msgs in self.queues.items()}
        self.queues = {stream: deque([msg for msg in msgs if msg['seq'] > seq], maxlen=msgs.maxlen) for stream, msgs in self.queues.items()}
        return res
