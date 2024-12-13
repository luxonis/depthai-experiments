import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended

import numpy as np

class ProcessDetections(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        
        self.crop_config = self.createOutput()
        self.num_frames_output = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            img_detections = self.detections_input.get()
            detections = img_detections.detections
            # w, h = img_detections.transformation.getSize()
            w, h = 1728, 960 
            
            num_frames = len(detections)
            
            for i, detection in enumerate(detections):
                cfg = dai.ImageManipConfigV2()
                detection: ImgDetectionExtended = detection
                rect = detection.rotated_rect
                rect = rect.denormalize(w, h)
                cfg.addCropRotatedRect(rect, normalizedCoords=False)
                cfg.setOutputSize(320, 48)
                cfg.setReusePreviousImage(False)
                
                self.crop_config.send(cfg)
            
            frame_message = dai.Buffer()
            frame_message.setData(np.array([num_frames], dtype=np.uint8))
            
            self.num_frames_output.send(frame_message)