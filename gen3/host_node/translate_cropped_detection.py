import depthai as dai
from depthai_nodes.ml.messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
)


class TranslateCroppedDetection(dai.node.HostNode):
    """Translates the detection (bbox, keypoints) coordinates from the cropped image to the original image."""

    def __init__(self):
        super().__init__()
        self._bbox_padding = 0.1
        self.initial_config = dai.ImageManipConfig()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, detection_nn: dai.Node.Output, cropped_nn: dai.Node.Output
    ) -> "TranslateCroppedDetection":
        self.link_args(detection_nn, cropped_nn)
        return self

    def process(self, detection_nn: dai.Buffer, cropped_nn: dai.Buffer):
        assert isinstance(detection_nn, (dai.ImgDetections, ImgDetectionsExtended))
        assert isinstance(
            cropped_nn,
            (dai.ImgDetections, ImgDetectionsExtended, Keypoints),
        )

        if len(detection_nn.detections) == 0:
            kpts_msg = Keypoints()
            kpts_msg.setTimestamp(cropped_nn.getTimestamp())
            kpts_msg.setSequenceNum(cropped_nn.getSequenceNum())
            self.output.send(kpts_msg)
            return
        elif len(detection_nn.detections) > 1:
            print(
                f"TranslateCroppedDetection expected 1 detection, got {len(detection_nn.detections)}, will use the one with highest confidence."
            )

        crop = sorted(
            detection_nn.detections, key=lambda d: d.confidence, reverse=True
        )[0]
        crop_box = (
            crop.xmin - self._bbox_padding,
            crop.ymin - self._bbox_padding,
            crop.xmax + self._bbox_padding,
            crop.ymax + self._bbox_padding,
        )
        msg = self._translate(crop_box, cropped_nn)

        self.output.send(msg)

    def _translate(
        self,
        crop_box: tuple[float, float, float, float],
        detection: ImgDetectionsExtended | dai.ImgDetections | Keypoints,
    ) -> ImgDetectionsExtended | dai.ImgDetections | Keypoints:
        if isinstance(detection, ImgDetectionsExtended):
            msg = ImgDetectionsExtended()
            msg.detections = [
                self._translate_detection(crop_box, det) for det in detection.detections
            ]
        elif isinstance(detection, dai.ImgDetections):
            msg = dai.ImgDetections()
            msg.detections = [
                self._translate_detection(crop_box, det) for det in detection.detections
            ]
        elif isinstance(detection, Keypoints):
            norm_kpts = [
                self._norm_point(crop_box, (i.x, i.y)) for i in detection.keypoints
            ]

            msg = Keypoints()
            msg.keypoints = [dai.Point3f(kpt[0], kpt[1], 0) for kpt in norm_kpts]
        msg.setSequenceNum(detection.getSequenceNum())
        msg.setTimestamp(detection.getTimestamp())
        return msg

    def _translate_detection(
        self,
        bbox: tuple[float, float, float, float],
        detection: ImgDetectionExtended | dai.ImgDetection,
    ) -> ImgDetectionExtended | dai.ImgDetection:
        if isinstance(detection, ImgDetectionExtended):
            img_det = ImgDetectionExtended()
            norm_kpts = [
                self._norm_point(bbox, (i[0], i[1])) for i in detection.keypoints
            ]
            img_det.keypoints = norm_kpts
        else:
            img_det = dai.ImgDetection()
        img_det.confidence = detection.confidence
        img_det.label = detection.label
        xmin, ymin = self._norm_point(bbox, (detection.xmin, detection.ymin))
        xmax, ymax = self._norm_point(bbox, (detection.xmax, detection.ymax))
        img_det.xmin = xmin
        img_det.ymin = ymin
        img_det.xmax = xmax
        img_det.ymax = ymax
        return img_det

    def _norm_point(
        self,
        bbox: tuple[float, float, float, float],
        point: tuple[float, float],
    ) -> tuple[float, float]:
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        return (
            (bbox[0] + (point[0] * bbox_w)),
            (bbox[1] + (point[1] * bbox_h)),
        )

    def set_bbox_padding(self, padding: float) -> None:
        self._bbox_padding = padding
