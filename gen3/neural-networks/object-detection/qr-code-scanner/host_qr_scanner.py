import cv2
import depthai as dai
import numpy as np
from pyzbar.pyzbar import decode


DECODE = True

class QRScanner(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "QRScanner":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        frame = preview.getCvFrame()

        for det in detections.detections:
            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
            if DECODE:
                text = self.decode(frame, bbox)
                cv2.putText(frame, text, (bbox[0] + 10, bbox[1] - 30)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
        
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def decode(self, frame, bbox):
        assert(DECODE)
        if bbox[1] == bbox[3] or bbox[0] == bbox[2]:
            print("Detection bbox is empty")
            return ""

        bbox = expandBbox(bbox, frame, 5)
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        data = decode(img)
        if data:
            text = data[0].data.decode("utf-8")
            print("Decoded text", text)
            return text
        else:
            print("Decoding failed")
            return ""


def expandBbox(bbox: np.ndarray, frame: np.ndarray, percentage: float) -> np.ndarray:
    bbox_copy = bbox.copy()
    pixels_expansion_0 = (bbox_copy[3] - bbox_copy[1]) * (percentage / 100)
    pixels_expansion_1 = (bbox_copy[2] - bbox_copy[0]) * (percentage / 100)
    bbox_copy[0] = max(0, bbox_copy[0] - pixels_expansion_1)
    bbox_copy[1] = max(0, bbox_copy[1] - pixels_expansion_0)
    bbox_copy[2] = min(frame.shape[1], bbox_copy[2] + pixels_expansion_1)
    bbox_copy[3] = min(frame.shape[0], bbox_copy[3] + pixels_expansion_0)
    return bbox_copy


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
