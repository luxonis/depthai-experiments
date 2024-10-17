import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
)


class NormalizeDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(
        self,
        frame: dai.Node.Output,
        nn: dai.Node.Output,
        manip_mode: dai.ImgResizeMode = dai.ImgResizeMode.STRETCH,
    ) -> "NormalizeDetections":
        self.manip_mode = manip_mode

        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, detections: dai.Buffer) -> None:
        frame = frame.getCvFrame()

        normalized_dets = self._normalize_detections(frame, detections)

        self.output.send(normalized_dets)

    def _normalize_detections(
        self,
        frame: np.ndarray,
        detections: dai.ImgDetections | ImgDetectionsExtended | Keypoints,
    ) -> dai.ImgDetections | ImgDetectionsExtended | Keypoints:
        if isinstance(detections, ImgDetectionsExtended):
            normalized_dets = ImgDetectionsExtended()
            normalized_dets.detections = [
                self._normalize_detection(frame, d) for d in detections.detections
            ]
        elif isinstance(detections, dai.ImgDetections):
            normalized_dets = dai.ImgDetections()
            normalized_dets.detections = [
                self._normalize_detection(frame, d) for d in detections.detections
            ]
        elif isinstance(detections, Keypoints):
            normalized_dets = Keypoints()
            kpts = [(i.x, i.y) for i in detections.keypoints]
            norm_kpts = self._normalize_kpts(frame, kpts)
            normalized_dets.keypoints = [dai.Point3f(i[0], i[1], 0) for i in norm_kpts]
        else:
            raise ValueError("Unknown detection type")
        normalized_dets.setSequenceNum(detections.getSequenceNum())
        normalized_dets.setTimestamp(detections.getTimestamp())
        return normalized_dets

    def _normalize_detection(
        self, frame: np.ndarray, detection: dai.ImgDetection | ImgDetectionExtended
    ) -> dai.ImgDetection | ImgDetectionExtended:
        if isinstance(detection, ImgDetectionExtended):
            img_det = ImgDetectionExtended()
            img_det.keypoints = self._normalize_kpts(frame, detection.keypoints)
        elif isinstance(detection, dai.ImgDetection):
            img_det = dai.ImgDetection()
        else:
            raise ValueError("Unknown detection type")

        norm_bbox = self._normalize_bbox(
            frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        )
        img_det.xmin = norm_bbox[0]
        img_det.ymin = norm_bbox[1]
        img_det.xmax = norm_bbox[2]
        img_det.ymax = norm_bbox[3]

        img_det.label = detection.label
        img_det.confidence = detection.confidence
        return img_det

    def _normalize_kpts(
        self,
        frame: np.ndarray,
        kpts: list[tuple[float, float] | tuple[float, float, float]],
    ) -> list[tuple[int, int] | tuple[int, int, float]]:
        return [self._norm_point((kpt[0], kpt[1]), frame) for kpt in kpts]

    def _normalize_bbox(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        return self._norm_point((bbox[0], bbox[1]), frame) + self._norm_point(
            (bbox[2], bbox[3]), frame
        )

    def _norm_point(
        self, kpt: tuple[float, float], frame: np.ndarray
    ) -> tuple[int, int]:
        # moves the point to equalize the crop
        if self.manip_mode == dai.ImgResizeMode.CROP:
            normVals = np.full(2, frame.shape[0])
            ret = (np.clip(np.array(kpt), 0, 1) * normVals).astype(int)
            ret[::2] += (frame.shape[1] - frame.shape[0]) // 2
            return tuple(ret.tolist())

        # moves the point to equalize the letterbox
        elif self.manip_mode == dai.ImgResizeMode.LETTERBOX:
            normVals = np.full(2, frame.shape[0])
            normVals[::2] = frame.shape[1]
            ret = (kpt[0], 0.5 + (kpt[1] - 0.5) * frame.shape[1] / frame.shape[0])
            return tuple((np.clip(np.array(ret), 0, 1) * normVals).astype(int).tolist())

        # moves the point based on the frame size
        else:
            normVals = np.full(2, frame.shape[0])
            normVals[::2] = frame.shape[1]
            ret = (np.clip(np.array(kpt), 0, 1) * normVals).astype(int).tolist()
            return tuple(ret)
