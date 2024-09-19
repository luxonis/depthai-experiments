import depthai as dai
import cv2
import numpy as np


class DrawDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        
        self.thickness = 2
        self.color = (255, 0, 0)
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])


    def build(self, frame: dai.Node.Output, nn: dai.Node.Output, label_map: list[str]) -> "DrawDetections":
        self.label_map = label_map
        
        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, in_frame : dai.ImgFrame, in_detections: dai.Buffer) -> None:
        frame = in_frame.getCvFrame()
        assert(isinstance(in_detections, dai.ImgDetections))
        out_frame = self.draw_detections(frame, in_detections.detections)
        
        img = self._create_img_frame(out_frame, dai.ImgFrame.Type.BGR888p)
        
        self.output.send(img)
        

    def draw_detections(self, frame : np.ndarray, detections : list[dai.ImgDetection]) -> np.ndarray:
        for detection in detections:
            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            bbox = (np.clip(np.array(bbox), 0, 1) * bbox).astype(int)
            cv2.putText(frame, self.label_map[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, self.thickness)

        return frame
    
    
    def set_color(self, color : tuple[int, int, int]) -> None:
        self.color = color
        
        
    def set_thickness(self, thickness : int) -> None:
        self.thickness = thickness
        
        
    def _create_img_frame(self, frame: np.ndarray, type : dai.ImgFrame.Type) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        return img_frame