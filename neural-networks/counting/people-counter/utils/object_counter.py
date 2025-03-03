import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class ObjectCount(dai.Buffer):
    def __init__(
        self, label_counts: dict[int, int], label_map: list[str] | None = None
    ) -> None:
        super().__init__(0)
        self._label_counts = label_counts
        self._label_map = label_map

    @property
    def label_counts(self):
        return self._label_counts

    @label_counts.setter
    def label_counts(self, value: dict[int, int]) -> None:
        self._label_counts = value

    def getTotalCount(self) -> int:
        return sum(self._label_counts.values())

    def getVisualizationMessage(self):
        label_counts = sorted(
            list(self._label_counts.items()), key=lambda x: x[0], reverse=True
        )
        if self._label_map:
            label_counts = [
                (self._label_map[label], count)
                for label, count in self._label_counts.items()
            ]

        img_annotations = dai.ImgAnnotations()
        img_annotations.setTimestamp(self.getTimestamp())
        img_annotations.setTimestampDevice(self.getTimestampDevice())
        img_annotations.setSequenceNum(self.getSequenceNum())

        line_size = 0.05
        annotation = dai.ImgAnnotation()
        for index, (label, count) in enumerate(label_counts):
            text_annotation = dai.TextAnnotation()
            text_annotation.text = f"{label}: {count}"
            text_annotation.fontSize = 15
            text_annotation.textColor = dai.Color(1.0, 1.0, 1.0)
            text_annotation.backgroundColor = dai.Color(0.5, 0.0, 1.0)
            text_annotation.position = dai.Point2f(0.05, 0.05 + line_size * index)

            annotation.texts.append(text_annotation)
        img_annotations.annotations.append(annotation)
        return img_annotations


class ObjectCounter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._confidence_threshold = 0.5
        self._label_map: list[str] | None = None
        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self._detection_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, nn: dai.Node.Output, label_map: list[str] | None = None
    ) -> "ObjectCounter":
        self.link_args(nn)
        self._label_map = label_map
        return self

    def process(self, detections: dai.Buffer) -> None:
        assert isinstance(detections, (dai.ImgDetections, ImgDetectionsExtended))

        label_counts = dict()

        filtered_detections = []

        for i in detections.detections:
            if i.confidence > self._confidence_threshold:
                filtered_detections.append(i)

                if i.label not in label_counts:
                    label_counts[i.label] = 0
                label_counts[i.label] += 1

        object_count = ObjectCount(label_counts, self._label_map)
        object_count.setTimestamp(detections.getTimestamp())
        object_count.setSequenceNum(detections.getSequenceNum())

        self._out.send(object_count)

        new_detections = type(detections)()
        new_detections.detections = filtered_detections
        new_detections.setTimestamp(detections.getTimestamp())
        new_detections.setSequenceNum(detections.getSequenceNum())

        self._detection_out.send(new_detections)

    def setConfidenceThreshold(self, confidence_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold

    @property
    def out(self) -> dai.Node.Output:
        return self._out

    @property
    def output(self) -> dai.Node.Output:
        return self._out

    @property
    def detection_out(self) -> dai.Node.Output:
        return self._detection_out
