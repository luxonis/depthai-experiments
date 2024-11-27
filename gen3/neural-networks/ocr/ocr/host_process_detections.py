import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended

import cv2
import numpy as np

class ProcessDetections(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput(blocking=False)
        self.frame = self.createInput(blocking=False)
        
        self.crop_config = self.createOutput()
        self.output_frame = self.createOutput()
        # self.output_rect = self.createOutput()

    def run(self) -> None:
        while self.isRunning():
            detections = self.detections_input.get()
            
            detections = detections.detections
            frame = self.frame.get()
            cvframe = frame.getCvFrame()    
            
            # print(f"Processing {len(detections)} detections.")
            for detection in detections:
                detection: ImgDetectionExtended = detection
                cfg = dai.ImageManipConfigV2()
                cfg.addCropRotatedRect(detection.rotated_rect, normalizedCoords=True)
                cfg.addResize(320, 48)
                cfg.setTimestamp(self.detections_input.get().getTimestamp())

                self.crop_config.send(cfg)
                self.output_frame.send(frame)
                

                rect = detection.rotated_rect   
                points = rect.getPoints()
                points = [[int(p.x * cvframe.shape[1]), int(p.y * cvframe.shape[0])] for p in points]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                cv2.polylines(cvframe, [points], isClosed=True, color=(0, 255, 0), thickness=3)
                
            cv2.imshow("cvframe", cvframe)
            cv2.waitKey(1)
            