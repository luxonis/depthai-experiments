import time


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0


    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

        
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)
