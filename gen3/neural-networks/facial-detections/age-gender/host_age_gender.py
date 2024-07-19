import depthai as dai
import numpy as np
import cv2
from detected_recognitions import DetectedRecognitions

class AgeGender(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, detections_recognitions: dai.Node.Output) -> "AgeGender":
        self.link_args(preview, detections_recognitions)
        self.sendProcessingToPipeline(True)
        return self

    # detections_recognitions is actually type DetectedRecognitions
    def process(self, preview: dai.ImgFrame, detections_recognitions: dai.Buffer) -> None:
        assert (isinstance(detections_recognitions, DetectedRecognitions))

        frame = preview.getCvFrame()
        detections = detections_recognitions.img_detections.detections
        recognitions = detections_recognitions.nn_data

        if detections and recognitions:
            for detection, recognition in zip(detections, recognitions):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                age = int((np.squeeze(np.array(recognition.getTensor("age_conv3").flatten()))) * 100)
                gender = np.squeeze(np.array(recognition.getTensor("prob").flatten()))
                gender_text = "female" if gender[0] > gender[1] else "male"

                y = (bbox[1] + bbox[3]) // 2
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z / 1000)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, gender_text, (bbox[0], y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
