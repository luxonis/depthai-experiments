from dataclasses import dataclass
from enum import Enum

import cv2
import depthai as dai
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detected_recognitions import DetectedRecognitions
from .labels import LABELS


@dataclass
class ColoredLabel:
    label: str
    color: tuple[int, int, int]


class TextPosition(Enum):
    TOP_LEFT = 0
    MID_LEFT = 1
    BOTTOM_LEFT = 2
    TOP_MID = 10
    MID_MID = 11
    BOTTOM_MID = 12
    TOP_RIGHT = 20
    MID_RIGHT = 21
    BOTTOM_RIGHT = 22


class VisualizedTracklets(dai.Tracklets):
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
                txt_annot.text = f"{LABELS[tracklet.srcImgDetection.label]} {round(tracklet.srcImgDetection.confidence * 100):<3}% ID: {tracklet.id}"
                txt_annot.position = dai.Point2f(tracklet.roi.x, tracklet.roi.y)
                txt_annot.textColor = dai.Color(0.0, 0.0, 0.0)
                txt_annot.backgroundColor = dai.Color(0.0, 1.0, 0.0)
                img_annotation.texts.append(txt_annot)

        img_annotations.annotations.append(img_annotation)
        return img_annotations


class DeepsortTracking(dai.node.HostNode):
    ALPHA = 0.15
    FONT_FACE = 0
    LINE_TYPE = cv2.LINE_AA
    FONT_SHADOW_COLOR = (0, 0, 0)
    FONT_COLOR = (255, 255, 255)
    TEXT_PADDING = 10

    def __init__(self) -> None:
        super().__init__()
        self._tracker = DeepSort(
            max_age=1000,
            nn_budget=None,
            embedder=None,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
        )
        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Tracklets, True)
            ]
        )

    def build(
        self,
        img_frames: dai.Node.Output,
        detected_recognitions: dai.Node.Output,
    ) -> "DeepsortTracking":
        self.link_args(img_frames, detected_recognitions)
        return self

    def process(
        self, img_frame: dai.ImgFrame, detected_recognitions: dai.Buffer
    ) -> None:
        frame = img_frame.getCvFrame()
        img_height, img_width, _ = frame.shape

        assert isinstance(detected_recognitions, DetectedRecognitions)
        detections = detected_recognitions.img_detections.detections
        recognitions = detected_recognitions.nn_data

        tracklets = VisualizedTracklets()

        if recognitions:
            object_tracks = self._tracker.iter(
                detections, recognitions, (img_width, img_height)
            )

            tracklet_list = []
            for track in object_tracks:
                if (
                    not track.is_confirmed()
                    or track.time_since_update > 1
                    or track.detection_id >= len(detections)
                    or track.detection_id < 0
                ):
                    continue
                tracklet = dai.Tracklet()
                det = detections[track.detection_id]
                tracklet.id = int(track.track_id)
                tracklet.age = track.age
                tracklet.label = det.label
                x, y, w, h = track.to_ltwh()
                rect = dai.Rect()
                rect.x = x / img_width
                rect.y = y / img_height
                rect.width = w / img_width
                rect.height = h / img_height
                tracklet.roi = rect
                tracklet.srcImgDetection = det
                tracklet.status = dai.Tracklet.TrackingStatus.TRACKED
                tracklet_list.append(tracklet)
            tracklets.tracklets = tracklet_list
        tracklets.setTimestamp(detected_recognitions.getTimestamp())
        tracklets.setSequenceNum(detected_recognitions.getSequenceNum())
        self._out.send(tracklets)

    @property
    def out(self) -> dai.Node.Output:
        return self._out
