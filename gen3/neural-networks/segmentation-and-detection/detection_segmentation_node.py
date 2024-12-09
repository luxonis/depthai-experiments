import depthai as dai
import numpy as np
import cv2
from depthai_nodes.ml.messages import ImgDetectionsExtended

class DetSegAnntotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        
        self.input_detections = self.createInput()
        self.input_frame = self.createInput()
        
        self.out = self.createOutput()
    
    def run(self):
        while self.isRunning():
            extended_detections = self.input_detections.get()
            frame = self.input_frame.get().getCvFrame()
            output_frame = dai.ImgFrame()
            
            if not isinstance(extended_detections, ImgDetectionsExtended):
               raise ValueError(f"Invalid input type. Expected ImgDetectionsExtended, got {type(extended_detections)}")

            label_mask = extended_detections.masks
            detections = extended_detections.detections
            if len(label_mask.shape) < 2:
                self.out.send(output_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888i))
                continue
            
            detection_labels = {idx: detection.label for idx, detection in enumerate(detections)}
            detection_labels[-1] = -1
            
            if len(detection_labels) > 0:
                label_mask = np.vectorize(lambda x: detection_labels.get(x, -1))(label_mask)
                        
            color_mask = label_mask.copy()
            color_mask[label_mask == -1] = 0
            color_mask = color_mask.astype(np.uint8)  
            color_mask = cv2.applyColorMap(color_mask, cv2.COLORMAP_HSV)
            color_mask[label_mask == -1] = frame[label_mask == -1]

            colored_frame = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
            
            self.out.send(output_frame.setCvFrame(colored_frame, dai.ImgFrame.Type.BGR888i))