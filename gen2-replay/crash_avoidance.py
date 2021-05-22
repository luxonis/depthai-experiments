import cv2
import math
from typing import no_type_check
import numpy as np
# Number of frames tracklet has to have "LOST" label to not display it
LOST_CNT = 20

class CrashAvoidance:
    def __init__(self, fps = None) -> None:
        self.tracklets_speed = []
        self.tracklets_roi = []
        self.lost_tracklets = []
        if fps is None: fps = 30.0
        self.time = 1.0 / fps

    def drawArrows(self, frame, detection):
        tracklets = self.__add_spatials_roi(detection)
        if len(tracklets["roi"]) < 15: return 0
        h = frame.shape[0]
        w = frame.shape[1]
        dir = self.__calc_direction(tracklets["roi"], 15, frame.shape)
        if dir is None: return frame
        p1, p2 = dir
        print(f"p1 {p1}, p2 {p2}")
        return cv2.arrowedLine(frame, p1, p2, color = (0, 0, 200), thickness=2, line_type=cv2.LINE_8, )

    def __calc_direction(self, roi_arr, num, shape):
        h = shape[0]
        w = shape[1]
        def get_centroid(rect):
            return (rect.x + rect.width / 2, rect.y + rect.height / 2)
        def subtract_centroids(p1, p2):
            x1,y1 = p1
            x2,y2 = p2
            return (x1 - x2, y1 - y2)

        arrLen = len(roi_arr)
        if arrLen < num: return 0
        # Last coordinates
        xSum = 0
        ySum = 0
        cnt = 0
        for i in range(num - 1):
            p1 = roi_arr[arrLen - i - 1]
            p2 = roi_arr[arrLen - i - 2]
            x, y = subtract_centroids(get_centroid(p1), get_centroid(p2))
            print(f"x {x}, y {y}")
            if x == 0.0 and y == 0.0: continue
            xSum += x
            ySum += y
            cnt += 1
        if cnt < 10: return None
        x = xSum / cnt
        y = ySum / cnt
        print(f"AVG x {x}, y {y}, num {num}")

        currX, currY = get_centroid(roi_arr[arrLen - 1])
        return ((int(currX * w),int(currY * h)), (int((currX + x * 10) * w),int((currY + y * 10) * h)))

    def calculate_speed(self, tracklet):
        def strP(p):
            return f"X: {p.x}, Y: {p.y}, Z: {p.z}"
        tracklets = self.__add_spatials(tracklet)
        # Not enough coords to calculate speed
        if len(tracklets["coords"]) < 15: return 0
        return self.__calc_speed_arr(tracklets["coords"], 15)

    def __calc_speed_arr(self, coords_arr, num):
        coordsLen = len(coords_arr)
        if coordsLen < num: return 0
        speeds = []
        # Last coordinates
        p1 = coords_arr[coordsLen - 1]
        for i in range(num - 1):
            p2 = coords_arr[coordsLen - 2 - i]
            dist = self.__calc_distance(p1,p2) / 1000 # In meters
            speed_mps = dist / (self.time * (i+1)) # Meters per second
            speed_kmph = speed_mps * 3.6 # KM per second
            speeds.append(speed_kmph)

        npSpeeds = np.array(speeds)
        npSpeeds = np.median(npSpeeds)
        return npSpeeds


    def __calc_distance(self, p1, p2):
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
    # Whether we should remove the lost tracklet, based on number of frames it has
    # been lost
    def remove_lost_tracklet(self, new_tracklet):
        saved = self.__get_or_add_lost(new_tracklet)
        if new_tracklet.status.name == "LOST":
            saved["lost_cnt"] += 1
        else: saved["lost_cnt"] = 0
        return LOST_CNT < saved["lost_cnt"]

    # Gets a tracklet from lost_tracklets arr or appends a new one
    def __get_or_add_lost(self, new_tracklet):
        for saved_tracklet in self.lost_tracklets:
            if saved_tracklet["id"] == new_tracklet.id:
                return saved_tracklet
        tracklet = {"id": new_tracklet.id, "lost_cnt": 0}
        self.lost_tracklets.append(tracklet)
        return tracklet

    def __add_spatials_speed(self, new_tracklet):
        for saved_tracklet in self.tracklets_speed:
            if saved_tracklet["id"] == new_tracklet.id:
                saved_tracklet["coords"].append(new_tracklet.spatialCoordinates)
                return saved_tracklet
        tracklet = {
            "id": new_tracklet.id,
            "coords": [new_tracklet.spatialCoordinates]
        }
        self.tracklets_speed.append(tracklet)
        return tracklet

    def __add_spatials_roi(self, new_tracklet):
        for saved_tracklet in self.tracklets_roi:
            if saved_tracklet["id"] == new_tracklet.id:
                saved_tracklet["roi"].append(new_tracklet.roi)
                return saved_tracklet
        tracklet = {
            "id": new_tracklet.id,
            "roi": [new_tracklet.roi]
        }
        self.tracklets_roi.append(tracklet)
        return tracklet