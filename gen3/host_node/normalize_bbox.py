import numpy as np
import depthai as dai
from enum import Enum


class MANIP_MODE(Enum):
    CROP, \
    LETTERBOX, \
    STRETCH = range(3)


class NormalizeBbox(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)])
        

    def build(self, frame : dai.Node.Output, nn : dai.Node.Output, manip_mode : MANIP_MODE = dai.ImgResizeMode.STRETCH) -> "NormalizeBbox":
        self.manip_mode = manip_mode

        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, frame : dai.ImgFrame, detections : dai.Buffer) -> None:
        frame = frame.getCvFrame()
        
        assert(isinstance(detections, dai.ImgDetections))
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