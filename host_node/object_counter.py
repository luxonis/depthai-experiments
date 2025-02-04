import depthai as dai


class ObjectCount(dai.Buffer):
    def __init__(self, label_counts: dict[int, int]) -> None:
        super().__init__(0)
        self._label_counts = label_counts

    @property
    def label_counts(self):
        return self._label_counts

    @label_counts.setter
    def label_counts(self, value: dict[int, int]) -> None:
        self._label_counts = value

    def getTotalCount(self) -> int:
        return sum(self._label_counts.values())


class ObjectCounter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._confidence_threshold = 0.5
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(self, nn: dai.Node.Output) -> "ObjectCounter":
        self.link_args(nn)
        return self

    def process(self, detections: dai.ImgDetections) -> None:
        label_counts = dict()
        for i in detections.detections:
            if i.confidence > self._confidence_threshold:
                if i.label not in label_counts:
                    label_counts[i.label] = 0
                label_counts[i.label] += 1

        object_count = ObjectCount(label_counts)
        object_count.setTimestamp(detections.getTimestamp())
        object_count.setSequenceNum(detections.getSequenceNum())
        self.output.send(object_count)

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold
