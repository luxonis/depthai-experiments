import cv2
import depthai as dai
import numpy as np
from host_node.detected_recognitions import DetectedRecognitions


emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

class DisplayEmotions(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, rgb : dai.Node.Output, detected_recognitions : dai.Node.Output, stereo : bool) -> "DisplayEmotions":
        self.stereo = stereo
        self.link_args(rgb, detected_recognitions)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgb_frame : dai.ImgFrame, detected_recognitions: dai.Buffer) -> None:
        frame = rgb_frame.getCvFrame()

        assert(isinstance(detected_recognitions, DetectedRecognitions))
        detections: dai.SpatialImgDetections = detected_recognitions.img_detections
        recognitions: list[dai.NNData] = detected_recognitions.nn_data

        dets = detections.detections

        for i, detection in enumerate(dets):
            bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            rec = recognitions[i]
            emotion_results = self.softmax(rec.getFirstTensor().astype(np.float16).flatten())
            emotion_name = emotions[np.argmax(emotion_results)]

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
            y = (bbox[1] + bbox[3]) // 2
            cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
            cv2.putText(frame, emotion_name, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
            if self.stereo:
                # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                cv2.putText(frame, coords, (bbox[0], y + 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                cv2.putText(frame, coords, (bbox[0], y + 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Camera", frame)     

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def frame_norm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)