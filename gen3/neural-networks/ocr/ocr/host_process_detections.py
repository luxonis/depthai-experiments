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
            frame = self.frame.get()
            img_detections = self.detections_input.get()
            
            detections = img_detections.detections
            cvframe = frame.getCvFrame()    
            w, h = img_detections.transformation.getSize()
            
            # print(f"Processing {len(detections)} detections.")
            for detection in detections:
                detection: ImgDetectionExtended = detection
                cfg = dai.ImageManipConfigV2()
                rect = detection.rotated_rect
                rect = rect.denormalize(w, h)
                cfg.addCropRotatedRect(rect, normalizedCoords=False)
                cfg.addResize(320, 48)
                cfg.setTimestamp(self.detections_input.get().getTimestamp())
                cfg.setTimestampDevice(self.detections_input.get().getTimestampDevice())
                
                self.crop_config.send(cfg)
                self.output_frame.send(frame)
            
            if len(detections) == 0:
                self.crop_config.send(dai.ImageManipConfigV2())
                black_frame = dai.ImgFrame()
                black_frame.setFrame(np.zeros((48, 320, 3), dtype=np.uint8))
                black_frame.setWidth(320)
                black_frame.setHeight(48)
                black_frame.setTimestamp(frame.getTimestamp())
                black_frame.setType(dai.ImgFrame.Type.BGR888i)
                
                self.output_frame.send(black_frame)

            #     points = rect.getPoints()
            #     points = [[int(p.x ), int(p.y )] for p in points]
            #     points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            #     cv2.circle(cvframe, (576, 320), 3, (0, 0, 255), 3)
            #     cv2.polylines(cvframe, [points], isClosed=True, color=(0, 255, 0), thickness=3)
                
            # cv2.imshow("cvframe", cvframe)
            # cv2.waitKey(1)
            