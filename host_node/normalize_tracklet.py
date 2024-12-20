from enum import Enum

import depthai as dai
import numpy as np


class MANIP_MODE(Enum):
    CROP, LETTERBOX, STRETCH = range(3)


class NormalizeTracklet(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Tracklets, True)
            ]
        )

    def build(
        self,
        frame: dai.Node.Output,
        tracklets: dai.Node.Output,
        manip_mode: MANIP_MODE = dai.ImgResizeMode.STRETCH,
    ) -> "NormalizeTracklet":
        self.manip_mode = manip_mode

        self.link_args(frame, tracklets)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, tracklets: dai.Buffer) -> None:
        frame = frame.getCvFrame()

        assert isinstance(tracklets, dai.Tracklets)
        normalized_tracklets = dai.Tracklets()
        normalized_tracklets.setTimestamp(tracklets.getTimestamp())
        normalized_tracklets.setSequenceNum(tracklets.getSequenceNum())
        norm_tracklets_list = []

        for tracklet in tracklets.tracklets:
            xmin, ymin = tracklet.roi.topLeft().x, tracklet.roi.topLeft().y
            xmax, ymax = tracklet.roi.bottomRight().x, tracklet.roi.bottomRight().y

            norm_det_bbox = self._frame_norm(
                frame,
                (
                    tracklet.srcImgDetection.xmin,
                    tracklet.srcImgDetection.ymin,
                    tracklet.srcImgDetection.xmax,
                    tracklet.srcImgDetection.ymax,
                ),
            )
            norm_det = dai.ImgDetection()
            norm_det.label = tracklet.srcImgDetection.label
            norm_det.confidence = tracklet.srcImgDetection.confidence
            norm_det.xmin = norm_det_bbox[0]
            norm_det.ymin = norm_det_bbox[1]
            norm_det.xmax = norm_det_bbox[2]
            norm_det.ymax = norm_det_bbox[3]

            norm_track_bbox = self._frame_norm(frame, (xmin, ymin, xmax, ymax))
            new_rect = dai.Rect(
                norm_track_bbox[0],
                norm_track_bbox[1],
                norm_track_bbox[2] - norm_track_bbox[0],
                norm_track_bbox[3] - norm_track_bbox[1],
            )
            norm_track = dai.Tracklet()
            norm_track.id = tracklet.id
            norm_track.label = tracklet.label
            norm_track.status = tracklet.status
            norm_track.spatialCoordinates = tracklet.spatialCoordinates
            norm_track.age = tracklet.age
            norm_track.srcImgDetection = norm_det
            norm_track.status = tracklet.status
            norm_track.roi = new_rect

            norm_tracklets_list.append(norm_track)

        normalized_tracklets.tracklets = norm_tracklets_list
        self.output.send(normalized_tracklets)

    def _frame_norm(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        # moves the bounding box to equalize the crop
        if self.manip_mode == dai.ImgResizeMode.CROP:
            normVals = np.full(4, frame.shape[0])
            ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
            ret[::2] += (frame.shape[1] - frame.shape[0]) // 2
            return ret

        # stretches the bounding box to equalize the letterbox
        elif self.manip_mode == dai.ImgResizeMode.LETTERBOX:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            bbox = (
                bbox[0],
                0.5 + (bbox[1] - 0.5) * frame.shape[1] / frame.shape[0],
                bbox[2],
                0.5 + (bbox[3] - 0.5) * frame.shape[1] / frame.shape[0],
            )
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        # resizes the bounding box based on the frame size
        else:
            normVals = np.full(4, frame.shape[0])
            normVals[::2] = frame.shape[1]
            ret = (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
            return ret
