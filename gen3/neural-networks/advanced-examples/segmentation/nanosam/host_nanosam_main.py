import numpy as np
import cv2
import depthai as dai

from decoder import ONNXDecoder
from utils import generate_overlay, frame_norm

class NanoSAM(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, detections: dai.Node.Output, nn: dai.Node.Output) -> "NanoSAM":
        self.link_args(preview, detections, nn)
        self.sendProcessingToPipeline(True)

        self.decoder = ONNXDecoder()
        return self

    def process(self, preview: dai.ImgFrame, detections: dai.ImgDetections, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()
        embeddings = np.array(nn.getFirstTensor()).flatten().reshape(1, 256, 64, 64)

        points, point_labels = [], []
        for detection in detections.detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

            points.append([bbox[0], bbox[1]])
            point_labels.append(2)
            points.append([bbox[2], bbox[3]])
            point_labels.append(3)

            mask, _, _ = self.decoder.predict(embeddings, np.array(points), np.array(point_labels))
            mask = (mask[:, :, 0] > 0).astype(np.uint8)
            frame = generate_overlay(frame, mask)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

