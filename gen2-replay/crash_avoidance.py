import math
from typing import no_type_check
import numpy as np
# Number of frames tracklet has to have "LOST" label to not display it
LOST_CNT = 20

class CrashAvoidance:
    def __init__(self) -> None:
        self.tracklets = []
        self.lost_tracklets = []
        self.FPS = 30.0
        self.time = 1.0 / self.FPS

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

    def __add_spatials(self, new_tracklet):
        for saved_tracklet in self.tracklets:
            if saved_tracklet["id"] == new_tracklet.id:
                saved_tracklet["coords"].append(new_tracklet.spatialCoordinates)
                return saved_tracklet
        tracklet = {
            "id": new_tracklet.id,
            "coords": [new_tracklet.spatialCoordinates]
        }
        self.tracklets.append(tracklet)
        return tracklet