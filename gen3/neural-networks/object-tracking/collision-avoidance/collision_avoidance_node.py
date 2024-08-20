import math
import numpy as np
import cv2
import depthai as dai

from tracker import Tracker
from collision_avoidance import CollisionAvoidance
from imutils.video import FPS


class CollisionAvoidanceNode(dai.node.HostNode):
    MAX_Z = 15
    MIN_Z = 0
    MAX_X = 1.3
    MIN_X = -0.5

    def __init__(self) -> None:
        super().__init__()
        self._tracker = Tracker()
        self._crash_avoidance = CollisionAvoidance()
        self._debug = True
        self._fps = FPS()
        self._fps.start()
        self._distance_bird_frame = self._make_bird_frame()


    def _make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((288, 100, 3), np.uint8)
        min_y = int((1 - (min_distance - self.MIN_Z) / (self.MAX_Z - self.MIN_Z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p),
            (center, frame.shape[0]),
            (0, max_p),
            (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame
        

    def build(self, img_frames: dai.Node.Output, nn: dai.Node.Output) -> "CollisionAvoidanceNode":
        self.link_args(img_frames, nn)
        self.sendProcessingToPipeline(True)
        return self
    

    def set_debug(self, debug: bool) -> None:
        self._debug = debug


    def process(self, img_frame: dai.ImgFrame, img_detections: dai.SpatialImgDetections) -> None:
        frame: np.ndarray = img_frame.getCvFrame()
        detections = img_detections.detections
        pts = [(item.spatialCoordinates.x, item.spatialCoordinates.z) for item in detections]
        tracker_objs = self._tracker.update(pts)
        crash_alert = self._crash_avoidance.parse(tracker_objs)
        if crash_alert:
            cv2.putText(frame, "DANGER!", (frame.shape[0] // 2, frame.shape[1] // 2 - 20), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 250))
        
        if not self._debug:
            return
        
        img_h = frame.shape[0]
        img_w = frame.shape[1]
        bird_frame = self._distance_bird_frame.copy()

        for detection in detections:
            left, top = int(detection.xmin * img_w), int(detection.ymin * img_h)
            right, bottom = int(detection.xmax * img_w), int(detection.ymax * img_h)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "x: {}".format(round(detection.spatialCoordinates.x, 1)), (left, top + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "y: {}".format(round(detection.spatialCoordinates.y, 1)), (left, top + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "z: {}".format(round(detection.spatialCoordinates.z, 1)), (left, top + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "conf: {}".format(round(detection.confidence, 1)), (left, top + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            left, right = self._calc_x(detection.spatialCoordinates.x, padding=2) 
            top, bottom = self._calc_z(detection.spatialCoordinates.z, padding=2)
            cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        for key in list(self._tracker.history.keys()):
            points = self._tracker.history[key]
            color = self._tracker.colors[key]

            draw_pts = np.array([(self._calc_x(pt[0]), self._calc_z(pt[1])) for pt in points])
            cv2.polylines(bird_frame, [draw_pts], False, color=color)

        numpy_horizontal = np.hstack((frame, bird_frame))
        cv2.imshow("DetectionFrame", numpy_horizontal)
        
        cv2.moveWindow("Bird", 1050, 200)
        cv2.moveWindow("Frame", 1370, 200)
        key = cv2.waitKey(1)

        if key == ord("q"):
            self.stopPipeline()


    def _calc_x(self, val: float, padding: int = 0) -> tuple[int, int] | int:
        norm = min(self.MAX_X, max(val, self.MIN_X))
        center = (norm - self.MIN_X) / (self.MAX_X - self.MIN_X) * self._distance_bird_frame.shape[1]
        if padding:
            bottom_x = max(center - padding, 0)
            top_x = min(center + padding, self._distance_bird_frame.shape[1])
            return int(bottom_x), int(top_x)
        else:
            return int(center)


    def _calc_z(self, val: float, padding: int = 0) -> tuple[int, int] | int:
        norm = min(self.MAX_Z, max(val, self.MIN_Z))
        center = (1 - (norm - self.MIN_Z) / (self.MAX_Z - self.MIN_Z)) * self._distance_bird_frame.shape[0]
        if padding:
            bottom_z = max(center - padding, 0)
            top_z = min(center + padding, self._distance_bird_frame.shape[0])
            return int(bottom_z), int(top_z)
        else:
            return int(center)
        

    