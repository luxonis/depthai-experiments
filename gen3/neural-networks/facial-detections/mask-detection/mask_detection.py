import depthai as dai


class MaskDetection(dai.node.HostNode):
    LABELS = ["No mask", "Mask"]

    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, ppe_nn: dai.Node.Output) -> "MaskDetection":
        self.link_args(ppe_nn)
        return self

    def process(self, img_detections: dai.ImgDetections) -> None:
        detections = img_detections.detections
        people_dets = [d for d in detections if d.label == 5]
        mask_dets = [d for d in detections if d.label == 1]
        no_mask_dets = [d for d in detections if d.label == 3]

        has_mask = [False for _ in range(len(people_dets))]
        for mask in mask_dets:
            for i, person in enumerate(people_dets):
                if self._is_inside_bbox(
                    self._det_to_bbox(person), self._det_to_bbox(mask)
                ):
                    has_mask[i] = True
        for no_mask in no_mask_dets:
            for i, person in enumerate(people_dets):
                if self._is_inside_bbox(
                    self._det_to_bbox(person), self._det_to_bbox(no_mask)
                ):
                    has_mask[i] = False

        img_dets = dai.ImgDetections()
        dets = []
        for has_mask, det in zip(has_mask, people_dets):
            new_det = dai.ImgDetection()
            new_det.label = int(has_mask)
            new_det.confidence = det.confidence
            new_det.xmin = det.xmin
            new_det.ymin = det.ymin
            new_det.xmax = det.xmax
            new_det.ymax = det.ymax
            dets.append(new_det)
        img_dets.detections = dets
        img_dets.setTimestamp(img_detections.getTimestamp())
        img_dets.setSequenceNum(img_detections.getSequenceNum())
        self.output.send(img_dets)

    def _is_inside_bbox(
        self,
        outter_bbox: tuple[float, float, float, float],
        inner_bbox: tuple[float, float, float, float],
    ) -> bool:
        return (
            outter_bbox[0] <= inner_bbox[0]
            and outter_bbox[1] <= inner_bbox[1]
            and outter_bbox[2] >= inner_bbox[2]
            and outter_bbox[3] >= inner_bbox[3]
        )

    def _det_to_bbox(
        self, detection: dai.ImgDetection
    ) -> tuple[float, float, float, float]:
        return (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
