import depthai as dai
from typing import List

THRESH_DIST_DELTA = 0.6

class PeopleTracker:
    def __init__(self) -> None:
        self.data = dict()
        self.counter = [0,0,0,0]

    mapping = ['left', 'right', 'up', 'down']
    def _print(self, direction):
        print(f"Person moved {self.mapping[direction]}")

    def _tracklet_removed(self, tracklet, coords2):
        coords1 = tracklet['coords']
        deltaX = coords2[0] - coords1[0]
        deltaY = coords2[1] - coords1[1]

        if abs(deltaX) > abs(deltaY) and abs(deltaX) > THRESH_DIST_DELTA:
            direction = 0 if 0 > deltaX else 1
            self.counter[direction] += 1
            self._print(direction)

        elif abs(deltaY) > abs(deltaX) and abs(deltaY) > THRESH_DIST_DELTA:
            direction = 2 if 0 > deltaY else 3
            self.counter[direction] += 1
            self._print(direction)

        # else: node.warn("Invalid movement")

    def _get_centroid(self, roi):
        x1 = roi.topLeft().x
        y1 = roi.topLeft().y
        x2 = roi.bottomRight().x
        y2 = roi.bottomRight().y
        return ((x2-x1)/2+x1, (y2-y1)/2+y1)

    def calculate_tracklet_movement(self, tracklets: dai.Tracklets) -> List[int]:
        for t in tracklets.tracklets:
            # If new tracklet, save its centroid
            if t.status == dai.Tracklet.TrackingStatus.NEW:
                self.data[str(t.id)] = {} # Reset
                self.data[str(t.id)]['coords'] = self._get_centroid(t.roi)
            elif t.status == dai.Tracklet.TrackingStatus.TRACKED:
                self.data[str(t.id)]['lostCnt'] = 0
            elif t.status == dai.Tracklet.TrackingStatus.LOST:
                self.data[str(t.id)]['lostCnt'] += 1
                # If tracklet has been "LOST" for more than 10 frames, remove it
                if 10 < self.data[str(t.id)]['lostCnt'] and "lost" not in self.data[str(t.id)]:
                    #print(f"Tracklet {t.id} lost: {self.data[str(t.id)]['lostCnt']}")
                    self._tracklet_removed(self.data[str(t.id)], self._get_centroid(t.roi))
                    self.data[str(t.id)]["lost"] = True
            elif (t.status == dai.Tracklet.TrackingStatus.REMOVED) and "lost" not in self.data[str(t.id)]:
                self._tracklet_removed(self.data[str(t.id)], self._get_centroid(t.roi))
            #print(f"Tracklet {t.id} removed")
        return self.counter