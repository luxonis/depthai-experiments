import math
import time
from collections import OrderedDict

import cv2
import numpy as np

from config import DEBUG


class CrashAvoidance:
    def __init__(self, calculated_entries=5, collision_trajectory_threshold=0.05, collision_time_to_impact=4):
        self.calculated_entries = calculated_entries
        self.collision_trajectory_threshold = collision_trajectory_threshold
        self.collision_time_to_impact = collision_time_to_impact
        self.entries = OrderedDict()

    def best_fit_slope_and_intercept(self, objectID):
        points = [item['value'] for item in self.entries[objectID]]
        xs = np.array([item[0] for item in points])
        zs = np.array([item[1] for item in points])

        m, b = np.polyfit(xs, zs, 1)

        return m, b

    def is_dangerous_trajectory(self, objectID):
        try:
            m, b = self.best_fit_slope_and_intercept(objectID)
        except ValueError:
            return False
        distance = abs(b) / math.sqrt(math.pow(m, 2) + 1)

        if DEBUG:
            image = np.ones((200, 200)) * 255
            cv2.line(image, (0, int(-100 * m + b)), (200, int(100 * m + b)), (0, 0, 0))
            cv2.putText(image, f"Distance: {round(distance, 2)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            if distance < self.collision_trajectory_threshold:
                cv2.putText(image, "DANGER", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.imshow(f"trajectory", image)
        return distance < self.collision_trajectory_threshold

    def is_impact_close(self, objectID):
        last, first = self.entries[objectID][0], self.entries[objectID][-1]
        x1, z1 = last['value']
        x2, z2 = first['value']
        lf_distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(z1 - z2, 2))
        if lf_distance == 0:
            return False
        timed = first['timestamp'] - last['timestamp']
        speed = lf_distance / timed  # m/s
        target_distance = math.sqrt(math.pow(x2, 2) + math.pow(z2, 2))
        tti = target_distance / speed

        if DEBUG:
            image = np.ones((200, 200)) * 255
            cv2.putText(image, f"LFD: {round(lf_distance, 2)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(image, f"SPD: {round(speed, 2)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(image, f"TDT: {round(target_distance, 2)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(image, f"TTI: {round(tti, 2)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            if tti < self.collision_trajectory_threshold:
                cv2.putText(image, "DANGER", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.imshow(f"impact", image)
        return tti < self.collision_time_to_impact

    def parse(self, tracker_objects):
        for key in list(self.entries.keys()):
            if key not in tracker_objects:
                del self.entries[key]

        for key in tracker_objects.keys():
            item = {
                'timestamp': time.time(),
                'value': tracker_objects[key]
            }
            if key not in self.entries:
                self.entries[key] = [item]
            else:
                self.entries[key] = (self.entries[key] + [item])[-self.calculated_entries:]

            if len(self.entries[key]) > 2 and self.is_dangerous_trajectory(key) and self.is_impact_close(key):
                return True
            else:
                return False



