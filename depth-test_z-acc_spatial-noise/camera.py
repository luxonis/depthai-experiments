import numpy as np
import open3d as o3d
import cv2
from typing import Optional, Tuple

class Camera:
    def __init__(self, name=""):
        self.image_frame: Optional[np.ndarray] = None
        self.image_visualization_frame: Optional[np.ndarray] = None
        self.depth_frame: Optional[np.ndarray] = None
        self.depth_visualization_frame: Optional[np.ndarray] = None
        self.point_cloud = o3d.geometry.PointCloud()
        self.ROI: Tuple[int, int, int, int] = (0, 0, 0, 0)

        self.window_name = f"Camera_{name}"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, lambda e, x, y, flags, param: self.on_mouse(e, x,y, flags, param))

        self.selecting_ROI = False

        self.extrinsic = np.eye(4)
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()

    def update(self):
        # capture image and depth frames
        pass

    def visualize_image_frame(self):
        if self.image_frame is not None:
            image_frame_roi = self.select_ROI(self.image_frame)
            self.image_visualization_frame = (self.image_frame*0.5 + image_frame_roi*0.5).astype(np.uint8)
            cv2.imshow(self.window_name, self.image_visualization_frame)

    def on_mouse(self, event, x, y, flags, param):
        x1, y1, x2, y2 = self.ROI
        if self.selecting_ROI:
            x2, y2 = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = x, y, x, y
            self.selecting_ROI = True
        elif event == cv2.EVENT_LBUTTONUP and self.selecting_ROI:
            self.selecting_ROI = False
            if abs(x1 - x2) * abs(y1 - y2) == 0:
                x1, y1, x2, y2 = (0, 0, self.image_frame.shape[1], self.image_frame.shape[0])
        
        self.ROI = (x1, y1, x2, y2)
        

    def select_ROI(self, frame: np.ndarray):
        x1, y1, x2, y2 = self.ROI
        ROI_sorted = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        x1, y1, x2, y2 = ROI_sorted
        ROI_mask = np.zeros_like(frame)
        ROI_mask[y1 : y2, x1 : x2] = 1
        
        return frame * ROI_mask


    def rgbd_to_point_cloud(self, downsample=False, remove_noise=False):
        image_frame = self.select_ROI(self.image_frame)
        depth_frame = self.select_ROI(self.depth_frame)

        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR))
        df = np.copy(depth_frame).astype(np.float32)
        depth_o3d = o3d.geometry.Image(df)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(image_frame.shape) != 3)
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic, self.extrinsic
        )

        if downsample:
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

        if remove_noise:
            point_cloud = point_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)[0]

        self.point_cloud.points = point_cloud.points
        self.point_cloud.colors = point_cloud.colors

        # correct upside down z axis
        R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.point_cloud.rotate(R_camera_to_world, center=(0, 0, 0))


        return self.point_cloud