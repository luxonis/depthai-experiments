import depthai as dai
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detected_recognitions import DetectedRecognitions
from .visualized_tracklets import VisualizedTracklets


class DeepsortTracking(dai.node.HostNode):
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
        assert isinstance(detected_recognitions, DetectedRecognitions)

        frame = img_frame.getCvFrame()
        img_height, img_width, _ = frame.shape

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
