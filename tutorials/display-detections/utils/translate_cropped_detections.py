import depthai as dai


class TranslateCroppedDetections(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(
        self,
        detections: dai.Node.Output,
        original_frame_size: tuple[int, int],
        crop_frame_size: tuple[int, int],
    ):
        self._original_frame_size = original_frame_size
        self._crop_frame_size = crop_frame_size
        self.link_args(detections)
        return self

    def process(self, detections: dai.ImgDetections):
        new_dets = dai.ImgDetections()
        new_dets_list = []
        for detection in detections.detections:
            new_det = dai.ImgDetection()
            new_det.label = detection.label
            new_det.confidence = detection.confidence

            new_det.xmin = (
                detection.xmin * self._crop_frame_size[0] / self._original_frame_size[0]
            )
            new_det.ymin = (
                detection.ymin * self._crop_frame_size[1] / self._original_frame_size[1]
            )
            new_det.xmax = (
                detection.xmax * self._crop_frame_size[0] / self._original_frame_size[0]
            )
            new_det.ymax = (
                detection.ymax * self._crop_frame_size[1] / self._original_frame_size[1]
            )
            new_dets_list.append(new_det)
        new_dets.detections = new_dets_list
        new_dets.setTimestamp(detections.getTimestamp())
        new_dets.setSequenceNum(detections.getSequenceNum())
        self.out.send(new_dets)

    @property
    def out(self):
        return self._out
