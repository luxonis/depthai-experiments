import depthai as dai
import cv2
import numpy as np
from enum import Enum


class MANIP_MODE(Enum):
    CROP, \
    LETTERBOX, \
    STRETCH = range(3)


class NormaliezeBbox(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)])
        

    def build(self, frame : dai.Node.Output, nn : dai.Node.Output, manip_mode : MANIP_MODE = dai.ImgResizeMode.CROP) -> "NormaliezeBbox":
        self.manip_mode = manip_mode

        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, frame : dai.ImgFrame, detections : dai.ImgDetections) -> None:
        frame = frame.getCvFrame()
        
        normalized_dets = dai.ImgDetections()
        normalized_dets.setTimestamp(detections.getTimestamp())
        normalized_dets.setSequenceNum(detections.getSequenceNum())
        dets = []
        
        for detection in detections.detections:
            bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            det = dai.ImgDetection()
            det.label = detection.label
            det.confidence = detection.confidence
            det.xmin = bbox[0]
            det.ymin = bbox[1]
            det.xmax = bbox[2]
            det.ymax = bbox[3]
            
            dets.append(det)
            
        normalized_dets.detections = dets
        self.output.send(normalized_dets)
    
    
    def _frame_norm(self, frame : np.ndarray, bbox : list[dai.ImgDetection]) -> np.ndarray:
        # moves the bounding box to equalize the crop
        if self.manip_mode == dai.ImgResizeMode.CROP:
            normVals = np.full(4, frame.shape[0])
            ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
            ret[::2] += (frame.shape[1] - frame.shape[0])//2
            return ret
        
        # stretches the bounding box to equalize the letterbox
        elif self.manip_mode == dai.ImgResizeMode.LETTERBOX:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            bbox = (bbox[0]
                    , 0.5 + (bbox[1]-0.5)*frame.shape[1]/frame.shape[0]
                    , bbox[2]
                    , 0.5 + (bbox[3]-0.5)*frame.shape[1]/frame.shape[0])
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        # resizes the bounding box based on the frame size
        else:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
            return ret



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


    def process(self, in_frame : dai.ImgFrame, in_detections: dai.ImgDetections) -> None:
        frame = in_frame.getCvFrame()
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