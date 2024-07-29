import numpy as np
import cv2
import depthai as dai

from detected_recognitions import DetectedRecognitions


class MaskDetection(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    
    def build(self, img_frames: dai.Node.Output, detected_recognitions: dai.Node.Output) -> "MaskDetection":
        self.link_args(img_frames, detected_recognitions)
        return self
    

    def process(self, img_frame: dai.ImgFrame, detected_recognitions: dai.Buffer) -> None:
        frame = img_frame.getCvFrame()
        assert(isinstance(detected_recognitions, DetectedRecognitions))
        detections = detected_recognitions.img_detections.detections

        for i, detection in enumerate(detections):
            bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            rec = detected_recognitions.nn_data[i].getFirstTensor()
            index = np.argmax(self._log_softmax(rec))
            text = "No Mask"
            color = (0,0,255) # Red
            if index == 1:
                text = "Mask"
                color = (0,255,0)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            y = (bbox[1] + bbox[3]) // 2
            cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
            cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
            if isinstance(detection, dai.SpatialImgDetection):
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    
    def _log_softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return np.log(e_x / e_x.sum())


    def _frame_norm(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)