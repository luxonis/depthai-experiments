import depthai as dai
import cv2
import numpy as np


class AddDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        
        self.thickness = 2
        self.color = (255, 0, 0)
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])


    def build(self, frame: dai.Node.Output, nn: dai.Node.Output, label_map: list[str]) -> "AddDetections":
        self.label_map = label_map
        
        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, in_frame : dai.ImgFrame, in_detections: dai.ImgDetections) -> None:
        frame = in_frame.getCvFrame()
        out_frame = self.add_detections(frame, in_detections.detections)
        
        img = dai.ImgFrame()
        img.setCvFrame(out_frame, dai.ImgFrame.Type.BGR888p)
        
        self.output.send(img)
        
        
    def frame_norm(self, frame : np.ndarray, bbox : list[dai.ImgDetection]) -> np.ndarray:
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals = (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
        norm_vals[::2] += (frame.shape[1] - frame.shape[0])//2
    
        return norm_vals


    def add_detections(self, frame : np.ndarray, detections : list[dai.ImgDetection]) -> np.ndarray:
        for detection in detections:
            bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, self.label_map[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, self.thickness)

        return frame
    
    
    def set_color(self, color : tuple[int, int, int]) -> None:
        self.color = color
        
        
    def set_thickness(self, thickness : int) -> None:
        self.thickness = thickness