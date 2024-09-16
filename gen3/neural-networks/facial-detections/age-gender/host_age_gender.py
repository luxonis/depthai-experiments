import depthai as dai
import numpy as np
import cv2
from detected_recognitions import DetectedRecognitions


class AgeGender(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self._draw_bbox = True


    def build(self, preview: dai.Node.Output, detections_recognitions: dai.Node.Output) -> "AgeGender":
        self.link_args(preview, detections_recognitions)    
        return self
    

    def setDrawBoundingBox(self, draw: bool) -> None:
        self._draw_bbox = draw


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

                if self._draw_bbox:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, gender_text, (bbox[0], y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output = dai.ImgFrame()
        output.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
        output.setTimestamp(preview.getTimestamp())
        output.setSequenceNum(preview.getSequenceNum())
        output.setTimestampDevice(preview.getTimestampDevice())
        self.output.send(output)


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
