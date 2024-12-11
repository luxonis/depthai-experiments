import depthai as dai
import time
import os

class SnapsProducer(dai.node.ThreadedHostNode):

    def __init__(self):
        super().__init__()
        
        self.rgb_frame: dai.Node.Output = None
        self.nn_output: dai.Node.Output = None
        self.confidence_threshold: float = 0.6
        self.time_interval: float = 60.0
        self.last_update = time.time()

        self.events_manager = dai.EventsManager()
        self.events_manager.setLogResponse(True)
        if os.getenv("DEPTHAI_HUB_URL") is not None:
            self.events_manager.setUrl(os.getenv("DEPTHAI_HUB_URL"))

    def build(self, 
        confidence_threshold: float = 0.6, 
        time_interval: float = 60.0) -> None:

        self.confidence_threshold = confidence_threshold
        self.time_interval: float = time_interval
        self.last_update = time.time()

        return self

    def run(self):
        
        while self.isRunning():
            rgb_frame = self.rgb_frame.get()
            nn_output = self.nn_output.get()
            
            for det in nn_output.detections:
                if det.confidence < self.confidence_threshold and time.time() > self.last_update + self.time_interval:
                    self.last_update = time.time()
                    print("----------------- EVENT SENT -----------------")
                    self.events_manager.sendSnap("rgb", rgb_frame, [], ["demo"], {"model": "cup-models"})