import depthai as dai
import numpy as np
from host_node.annotation_builder import AnnotationBuilder
from pyzbar import pyzbar

DECODE = True


class QRDecoder(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)
            ]
        )

    def build(self, frame: dai.Node.Output, nn: dai.Node.Output) -> "QRDecoder":
        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        img = frame.getCvFrame()

        annotation_builder = AnnotationBuilder()
        for det in detections.detections:
            if DECODE:
                bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
                text = self.decode(img, bbox)
                annotation_builder.draw_text(
                    text, (bbox[0], bbox[1]), (255, 255, 255, 1), (0, 0, 255, 1), 24
                )
        annotation = annotation_builder.build(
            detections.getTimestamp(), detections.getSequenceNum()
        )
        self.output.send(annotation)

    def _frame_norm(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def _expand_bbox(
        self, bbox: np.ndarray, frame: np.ndarray, percentage: float
    ) -> np.ndarray:
        bbox_copy = bbox.copy()
        pixels_expansion_0 = (bbox_copy[3] - bbox_copy[1]) * (percentage / 100)
        pixels_expansion_1 = (bbox_copy[2] - bbox_copy[0]) * (percentage / 100)
        bbox_copy[0] = max(0, bbox_copy[0] - pixels_expansion_1)
        bbox_copy[1] = max(0, bbox_copy[1] - pixels_expansion_0)
        bbox_copy[2] = min(frame.shape[1], bbox_copy[2] + pixels_expansion_1)
        bbox_copy[3] = min(frame.shape[0], bbox_copy[3] + pixels_expansion_0)
        return bbox_copy

    def decode(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
        assert DECODE
        norm_bbox = self._frame_norm(frame, bbox)
        if norm_bbox[1] == norm_bbox[3] or norm_bbox[0] == norm_bbox[2]:
            print("Detection bbox is empty")
            return ""

        norm_bbox = self._expand_bbox(norm_bbox, frame, 5)
        img = frame[norm_bbox[1] : norm_bbox[3], norm_bbox[0] : norm_bbox[2]]

        data = pyzbar.decode(img)
        if data:
            text = data[0].data.decode("utf-8")
            print("Decoded text", text)
            return text
        else:
            print("Decoding failed")
            return ""
