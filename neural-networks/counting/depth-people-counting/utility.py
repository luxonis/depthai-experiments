import depthai as dai
import cv2

DETECTION_ROI = (
    200,
    300,
    1000,
    700,
)  # Specific to `depth-person-counting-01` recording
THRESH_DIST_DELTA = 0.5


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(
            frame, text, coords, self.text_type, 1.3, self.bg_color, 5, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 1.3, self.color, 2, self.line_type
        )
        return frame

    def rectangle(self, frame, topLeft, bottomRight, size=1.0):
        cv2.rectangle(frame, topLeft, bottomRight, self.bg_color, int(size * 4))
        cv2.rectangle(frame, topLeft, bottomRight, self.color, int(size))
        return frame


class PeopleCounter:
    def __init__(self):
        self.tracking = {}
        self.lost_cnt = {}
        self.people_counter = [0, 0, 0, 0]  # Up, Down, Left, Right

    def __str__(self) -> str:
        return f"Left: {self.people_counter[2]}, Right: {self.people_counter[3]}"

    def tracklet_removed(self, coords1, coords2):
        dx = coords2[0] - coords1[0]

        if THRESH_DIST_DELTA < abs(dx):
            self.people_counter[2 if 0 > dx else 3] += 1
            direction = "left" if 0 > dx else "right"
            print(f"Person moved {direction}")

    def get_centroid(self, roi):
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y
        return (x2 + x1) / 2, (y2 + y1) / 2

    def new_tracklets(self, tracklets):
        for t in tracklets:
            # If new tracklet, save its centroid
            if t.status == dai.Tracklet.TrackingStatus.NEW:
                self.tracking[str(t.id)] = self.get_centroid(t.roi)
                self.lost_cnt[str(t.id)] = 0
            elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.lost_cnt[str(t.id)] = 0
            elif t.status == dai.Tracklet.TrackingStatus.LOST:
                self.lost_cnt[str(t.id)] += 1
                # Tracklet has been lost for too long
                if 10 < self.lost_cnt[str(t.id)]:
                    self.lost_cnt[str(t.id)] = -999
                    self.tracklet_removed(
                        self.tracking[str(t.id)], self.get_centroid(t.roi)
                    )
            elif t.status == dai.Tracklet.TrackingStatus.REMOVED:
                if 0 <= self.lost_cnt[str(t.id)]:
                    self.lost_cnt[str(t.id)] = -999
                    self.tracklet_removed(
                        self.tracking[str(t.id)], self.get_centroid(t.roi)
                    )
