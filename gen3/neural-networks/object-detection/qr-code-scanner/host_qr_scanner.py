import cv2
import depthai as dai
import numpy as np

DECODE = True

class QRScanner(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.detector = cv2.QRCodeDetector() if DECODE else None

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "QRScanner":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        frame = preview.getCvFrame()

        for det in detections.detections:
            expandDetection(det, 2)
            bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
            if DECODE:
                text = self.decode(frame, bbox)
                cv2.putText(frame, text, (bbox[0] + 10, bbox[1] - 30)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def decode(self, frame, bbox):
        assert(DECODE)

        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        data, vertices_array, binary_qrcode = self.detector.detectAndDecode(img)
        if data:
            print("Decoded text", data)
            return data
        else:
            print("Decoding failed")
            return ""

def expandDetection(det, percent):
    percent /= 100
    det.xmin -= percent
    det.ymin -= percent
    det.xmax += percent
    det.ymax += percent
    if det.xmin < 0: det.xmin = 0
    if det.ymin < 0: det.ymin = 0
    if det.xmax > 1: det.xmax = 1
    if det.ymax > 1: det.ymax = 1

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
