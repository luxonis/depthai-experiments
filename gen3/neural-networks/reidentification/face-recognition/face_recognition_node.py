import cv2
import depthai as dai
import numpy as np

from face_recognition import FaceRecognition
from text_helper import TextHelper
from detected_recognitions import DetectedRecognitions


class FaceRecognitionNode(dai.node.HostNode):
    def __init__(self) -> None:
        self._text_helper = TextHelper()
        super().__init__()


    def build(self, img_output: dai.Node.Output, detected_recognitions: dai.Node.Output, face_recognition: FaceRecognition) -> "FaceRecognitionNode":
        self.link_args(img_output, detected_recognitions)
        self.sendProcessingToPipeline(True)
        self._face_recognition = face_recognition
        return self
    

    def process(self, color: dai.ImgFrame, detected_recognitions) -> None:
            assert(isinstance(detected_recognitions, DetectedRecognitions))
            detections: dai.ImgDetections = detected_recognitions.img_detections
            recognitions: list[dai.NNData] = detected_recognitions.nn_data

            frame = color.getCvFrame()
            dets = detections.detections

            for i, detection in enumerate(dets):
                bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                firstLayer = recognitions[i].getAllLayerNames()[0]
                features = recognitions[i].getTensor(firstLayer).astype(np.float16).flatten()
                conf, name = self._face_recognition.new_recognition(features)
                self._text_helper.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10,bbox[1] + 35))

            cv2.imshow("color", cv2.resize(frame, (800,800)))

            if cv2.waitKey(1) == ord('q'):
                self.stopPipeline()                
            

    def _frame_norm(self, frame: np.ndarray, bbox: tuple):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)