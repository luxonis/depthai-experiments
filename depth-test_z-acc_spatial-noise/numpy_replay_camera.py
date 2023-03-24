import depthai as dai
import cv2
import numpy as np
import open3d as o3d
from typing import List
import config
from camera import Camera
from host_sync import HostSync
from pathlib import Path

class NumpyReplayCamera(Camera):
    def __init__(self, depth_recording_path: Path, color_recording_path: Path, calibration_path: Path):
        super().__init__(name="OAK")
        self.depth_recording_path = depth_recording_path
        self.color_recording_path = color_recording_path
        self.calibration_path = calibration_path

        self.depth_frames = np.load(depth_recording_path)
        self.color_frames = np.load(color_recording_path).astype(np.uint8)
        # use only last 3 frames
        self.depth_frames = self.depth_frames[-3:]
        self.color_frames = self.color_frames[-3:]
        self.frame_index = 0
        self.no_frames = min(len(self.depth_frames), len(self.color_frames))
        self.image_size = self.depth_frames[0].shape

        self._load_calibration()

    def __del__(self):
        print("=== Closed " + self.depth_recording_path)

    def _load_calibration(self):
        calibration = dai.CalibrationHandler(self.calibration_path)
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT, 
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )

        self.focal_length = self.intrinsics[0][0] # in pixels
        self.stereoscopic_baseline = calibration.getBaselineDistance()/100 # in m

        self.extrinsic = np.eye(4)


    def update(self):
        self.depth_frame = self.depth_frames[self.frame_index]
        self.image_frame = self.color_frames[self.frame_index]

        if self.frame_index < self.no_frames - 1:
            self.frame_index += 1
        else:
            self.frame_index = 0
        
        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)

        self.visualize_image_frame()
        self.rgbd_to_point_cloud()