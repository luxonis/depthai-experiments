import depthai as dai


class VisualizedTracklets(dai.Tracklets):
    def __init__(self):
        super().__init__()
        self._labels = None

    def setLabels(self, labels: list[str]):
        self._labels = labels

    def getVisualizationMessage(self):
        img_annotations = dai.ImgAnnotations()
        img_annotations.setTimestamp(self.getTimestamp())
        img_annotations.setSequenceNum(self.getSequenceNum())
        img_annotation = dai.ImgAnnotation()
        for tracklet in self.tracklets:
            if tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                pts_annot = dai.PointsAnnotation()
                pts_annot.outlineColor = dai.Color(0.0, 1.0, 0.0)
                pts_annot.points.append(dai.Point2f(tracklet.roi.x, tracklet.roi.y))
                pts_annot.points.append(
                    dai.Point2f(tracklet.roi.x + tracklet.roi.width, tracklet.roi.y)
                )
                pts_annot.points.append(
                    dai.Point2f(
                        tracklet.roi.x + tracklet.roi.width,
                        tracklet.roi.y + tracklet.roi.height,
                    )
                )
                pts_annot.points.append(
                    dai.Point2f(tracklet.roi.x, tracklet.roi.y + tracklet.roi.height)
                )
                pts_annot.type = dai.PointsAnnotationType.LINE_LOOP
                pts_annot.thickness = 2
                img_annotation.points.append(pts_annot)
                txt_annot = dai.TextAnnotation()
                txt_annot.fontSize = 25
                if self._labels:
                    label_txt = self._labels[tracklet.srcImgDetection.label]
                else:
                    label_txt = tracklet.srcImgDetection.label
                txt_annot.text = f"{label_txt} {round(tracklet.srcImgDetection.confidence * 100):<3}% ID: {tracklet.id}"
                txt_annot.position = dai.Point2f(tracklet.roi.x, tracklet.roi.y)
                txt_annot.textColor = dai.Color(0.0, 0.0, 0.0)
                txt_annot.backgroundColor = dai.Color(0.0, 1.0, 0.0)
                img_annotation.texts.append(txt_annot)

        img_annotations.annotations.append(img_annotation)
        return img_annotations


class TrackletVisualizer(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, tracklets: dai.Node.Output, labels: list[str]
    ) -> "TrackletVisualizer":
        self._labels = labels
        self.link_args(tracklets)
        return self

    def process(self, tracklets: dai.Tracklets) -> None:
        vis_tracklets = VisualizedTracklets()
        vis_tracklets.tracklets = tracklets.tracklets
        vis_tracklets.setLabels(self._labels)
        vis_tracklets.setTimestamp(tracklets.getTimestamp())
        vis_tracklets.setSequenceNum(tracklets.getSequenceNum())
        self._out.send(vis_tracklets)

    @property
    def out(self) -> dai.Node.Output:
        return self._out
