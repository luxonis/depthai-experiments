import depthai as dai
from typing import List
from depthai_nodes.utils import AnnotationHelper
from depthai_nodes import SECONDARY_COLOR


class VisualizedTracklets(dai.Tracklets):
    def __init__(self):
        super().__init__()
        self._labels = None

    def setLabels(self, labels: List[str]):
        self._labels = labels

    def getVisualizationMessage(self):
        annotation_helper = AnnotationHelper()
        # img_annotations = dai.ImgAnnotations()
        # img_annotations.setTimestamp(self.getTimestamp())
        # img_annotations.setSequenceNum(self.getSequenceNum())
        # img_annotation = dai.ImgAnnotation()
        for tracklet in self.tracklets:
            if tracklet.status == dai.Tracklet.TrackingStatus.TRACKED:
                annotation_helper.draw_rectangle(
                    top_left=(tracklet.roi.x, tracklet.roi.y),
                    bottom_right=(
                        tracklet.roi.x + tracklet.roi.width,
                        tracklet.roi.y + tracklet.roi.height,
                    ),
                )
                # pts_annot = dai.PointsAnnotation()
                # pts_annot.outlineColor = dai.Color(0.0, 1.0, 0.0)
                # pts_annot.points.append(dai.Point2f(tracklet.roi.x, tracklet.roi.y))
                # pts_annot.points.append(
                #     dai.Point2f(tracklet.roi.x + tracklet.roi.width, tracklet.roi.y)
                # )
                # pts_annot.points.append(
                #     dai.Point2f(
                #         tracklet.roi.x + tracklet.roi.width,
                #         tracklet.roi.y + tracklet.roi.height,
                #     )
                # )
                # pts_annot.points.append(
                #     dai.Point2f(tracklet.roi.x, tracklet.roi.y + tracklet.roi.height)
                # )
                # pts_annot.type = dai.PointsAnnotationType.LINE_LOOP
                # pts_annot.thickness = 2
                # img_annotation.points.append(pts_annot)

                # txt_annot = dai.TextAnnotation()
                # txt_annot.fontSize = 25
                if self._labels:
                    label_txt = self._labels[tracklet.srcImgDetection.label]
                else:
                    label_txt = tracklet.srcImgDetection.label

                annotation_helper.draw_text(
                    text=f"{label_txt} {round(tracklet.srcImgDetection.confidence * 100):<3}% ID: {tracklet.id}",
                    position=(tracklet.roi.x, tracklet.roi.y + 0.025),
                    size=25,
                    color=SECONDARY_COLOR,
                )

        annotations = annotation_helper.build(
            timestamp=self.getTimestamp(),
            sequence_num=self.getSequenceNum(),
        )

        return annotations
