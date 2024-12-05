import depthai as dai
import numpy as np
import cv2
from depthai_nodes.ml.messages import SegmentationMask

class SegAnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        
        self.input_segmentation = self.createInput()
        self.input_frame = self.createInput()
        
        self.out = self.createOutput()
    
    def run(self):
        while self.isRunning():
            mask = self.input_segmentation.get()
            frame = self.input_frame.get().getCvFrame()
            output_frame = dai.ImgFrame()
            
            if not isinstance(mask, SegmentationMask):
               raise ValueError(f"Invalid input type. Expected SegmentationMask, got {type(mask)}")
           
            mask = mask.mask
            unique_values = np.unique(mask[mask >= 0])
            scaled_mask = np.zeros_like(mask, dtype=np.uint8)

            if unique_values.size != 0:

                min_val, max_val = unique_values.min(), unique_values.max()

                if min_val == max_val:
                    scaled_mask = np.ones_like(mask, dtype=np.uint8) * 255
                else:
                    scaled_mask = ((mask - min_val) / (max_val - min_val) * 255).astype(
                        np.uint8
                    )
                scaled_mask[mask == -1] = 0
            colored_mask = cv2.applyColorMap(scaled_mask, cv2.COLORMAP_RAINBOW)
            colored_mask[mask == -1] = [0, 0, 0]
            
            colored_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)
            
            self.out.send(output_frame.setCvFrame(colored_frame, dai.ImgFrame.Type.BGR888i))
            