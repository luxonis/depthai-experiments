import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended
from typing import List

import numpy as np

class ProcessDetections(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        
        self.config_output = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            img_detections = self.detections_input.get()
            detections = img_detections.detections
            # w, h = img_detections.transformation.getSize()
            w, h = 1728, 960 
            
            configs_message = dai.MessageGroup()
            for i, detection in enumerate(detections):
                cfg = dai.ImageManipConfigV2()
                detection: ImgDetectionExtended = detection
                rect = detection.rotated_rect
                rect = rect.denormalize(w, h)
                cfg.addCropRotatedRect(rect, normalizedCoords=False)
                cfg.setOutputSize(320, 48)
                cfg.setReusePreviousImage(False)
                cfg.setTimestamp(img_detections.getTimestamp())
                
                configs_message[str(i+100)] = cfg
                            
            configs_message.setTimestamp(img_detections.getTimestamp())
            self.config_output.send(configs_message)
