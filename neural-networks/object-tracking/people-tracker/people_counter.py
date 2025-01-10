import cv2
import depthai as dai
import numpy as np


class PeopleCounter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.tracking_data = {}
        # Y axis (up/down), X axis (left/right)
        self.counter = {"up": 0, "down": 0, "left": 0, "right": 0}
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, preview: dai.Node.Output, tracklets: dai.Node.Output, threshold: float
    ) -> "PeopleCounter":
        self.link_args(preview, tracklets)

        self._threshold = threshold
        return self

    def process(self, preview: dai.ImgFrame, tracklets: dai.Tracklets) -> None:
        img = preview.getCvFrame().copy()
        self.update(tracklets)
        img = self.draw_counter_text(img)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(preview.getTimestamp())
        img_frame.setSequenceNum(preview.getSequenceNum())

        self.output.send(img_frame)

    def update(self, tracklets: dai.Tracklets) -> None:
        for t in tracklets.tracklets:
            id = str(t.id)
            centroid = get_centroid(t.roi)

            # New tracklet, save its centroid
            if t.status == dai.Tracklet.TrackingStatus.NEW:
                self.tracking_data[id] = {}  # Reset
                self.tracking_data[id]["coords"] = centroid

            elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.tracking_data[id]["lost_count"] = 0

            elif t.status == dai.Tracklet.TrackingStatus.LOST:
                self.tracking_data[id]["lost_count"] += 1

                # Removes tracklet that has been lost for more than 10 frames
                if (
                    self.tracking_data[id]["lost_count"] > 10
                    and "lost" not in self.tracking_data[id]
                ):
                    self.terminate_tracklet(id, centroid)
                    self.tracking_data[id]["lost"] = True

            elif (
                t.status == dai.Tracklet.TrackingStatus.REMOVED
            ) and "lost" not in self.tracking_data[id]:
                self.terminate_tracklet(id, centroid)

    def draw_counter_text(self, frame: np.ndarray):
        counter_text = f"Up: {self.counter['up']}, Down: {self.counter['down']}, Left: {self.counter['left']}, Right: {self.counter['right']}"
        cv2.putText(
            frame, counter_text, (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255)
        )
        return frame

    def terminate_tracklet(self, id, coords_end):
        coords_start = self.tracking_data[id]["coords"]

        dx = coords_end[0] - coords_start[0]
        dy = coords_end[1] - coords_start[1]

        if abs(dx) > abs(dy) and abs(dx) > self._threshold:
            direction = "left" if dx < 0 else "right"
            self.counter[direction] += 1
            print(f"Person moved {direction}")

        elif abs(dy) > abs(dx) and abs(dy) > self._threshold:
            direction = "up" if dy < 0 else "down"
            self.counter[direction] += 1
            print(f"Person moved {direction}")


def get_centroid(roi):
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    return (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1
