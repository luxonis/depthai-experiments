import time
import numpy as np
import cv2
import depthai as dai

from trackable_object import TrackableObject


class CumulativeObjectCounting(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._start_time = time.monotonic()
        self._frame_count = 0
        self._axis = True
        self._roi_position = 0.5
        self._trackable_objects: dict[int, TrackableObject] = {}
        self._counter = [0, 0, 0, 0]
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    
    def build(self, img_frames: dai.Node.Output, tracklets: dai.Node.Output) -> "CumulativeObjectCounting":
        self.link_args(img_frames, tracklets)
        self.sendProcessingToPipeline(True)
        return self
    

    def set_axis(self, axis: bool) -> None:
        self._axis = axis
    

    def set_roi_position(self, roi_position: float) -> None:
        self._roi_position = roi_position
    
    
    def process(self, img_frame: dai.Buffer, tracklets: dai.Tracklets) -> None:
        assert(isinstance(img_frame, dai.ImgFrame))
        frame = img_frame.getCvFrame()

        self._draw_fps(frame)
        self._frame_count += 1

        height, width = frame.shape[0], frame.shape[1]

        for t in tracklets.tracklets:
            to = self._trackable_objects.get(t.id, None)
            centroid = self._calculate_centroid(height, width, t)

            if t.status == dai.Tracklet.TrackingStatus.NEW:
                to = TrackableObject(t.id, centroid)
            elif isinstance(to, TrackableObject):
                self._update_counter(height, width, to, centroid)
                to.centroids.append(centroid)

            self._trackable_objects[t.id] = to

            if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                self._draw_tracklet(frame, t, centroid)

        self._draw_roi_line(frame, height, width)
        self._draw_count_and_status(frame)

        out_frame = dai.ImgFrame()
        out_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
        self.output.send(out_frame)


    def _draw_fps(self, frame: np.ndarray) -> None:
        cv2.putText(frame, "NN fps: {:.2f}".format(self._frame_count / (time.monotonic() - self._start_time)),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))


    def _calculate_centroid(self, height: int, width: int, t: dai.Tracklet) -> tuple[int, int]:
        roi = t.roi.denormalize(width, height)
        x1 = int(roi.topLeft().x)
        y1 = int(roi.topLeft().y)
        x2 = int(roi.bottomRight().x)
        y2 = int(roi.bottomRight().y)
        centroid = (int((x2-x1)/2+x1), int((y2-y1)/2+y1))
        return centroid
    

    def _update_counter(self, height: int, width: int, to: TrackableObject, centroid: tuple[int, int]) -> None:
        if self._axis and not to.counted:
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)

            if centroid[0] > self._roi_position*width and direction > 0 and np.mean(x) < self._roi_position*width:
                self._counter[1] += 1
                to.counted = True
            elif centroid[0] < self._roi_position*width and direction < 0 and np.mean(x) > self._roi_position*width:
                self._counter[0] += 1
                to.counted = True

        elif not self._axis and not to.counted:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)

            if centroid[1] > self._roi_position*height and direction > 0 and np.mean(y) < self._roi_position*height:
                self._counter[3] += 1
                to.counted = True
            elif centroid[1] < self._roi_position*height and direction < 0 and np.mean(y) > self._roi_position*height:
                self._counter[2] += 1
                to.counted = True


    def _draw_tracklet(self, frame: np.ndarray, t: dai.Tracklet, centroid: tuple[int, int]) -> None:
        text = "ID {}".format(t.id)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(
                    frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


    def _draw_roi_line(self, frame: np.ndarray, height: int, width: int) -> None:
        if self._axis:
            cv2.line(frame, (int(self._roi_position*width), 0),
                        (int(self._roi_position*width), height), (0xFF, 0, 0), 5)
        else:
            cv2.line(frame, (0, int(self._roi_position*height)),
                        (width, int(self._roi_position*height)), (0xFF, 0, 0), 5)
            
    
    def _draw_count_and_status(self, frame: np.ndarray) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self._axis:
            cv2.putText(frame, f'Left: {self._counter[0]}; Right: {self._counter[1]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(frame, f'Up: {self._counter[2]}; Down: {self._counter[3]}', (
                10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
