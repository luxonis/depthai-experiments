import pyzed.sl as sl
import cv2
from camera import Camera
import time
import open3d as o3d
import numpy as np

# Create a Camera object

class ZedCamera(Camera):
    def __init__(self):
        super().__init__(name="Zed")

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("can't open Zed camera")

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.enable_fill_mode = True
        self._load_calibration()

    def __del__(self):
        self.zed.close()


    def _load_calibration(self):
        intrinsic_parameters = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.focal_length = intrinsic_parameters.fx
        self.stereoscopic_baseline = 0.12
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.zed.get_camera_information().camera_configuration.resolution.width,
            self.zed.get_camera_information().camera_configuration.resolution.height,
            intrinsic_parameters.fx, intrinsic_parameters.fy, intrinsic_parameters.cx, intrinsic_parameters.cy
        )
    def update(self,key = None):
        image_size = sl.Resolution(int(self.zed.get_camera_information().camera_configuration.resolution.width ),
                                   int(self.zed.get_camera_information().camera_configuration.resolution.height ))
        left_image = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        depth_map = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT, resolution=image_size)
            self.image_frame  = left_image.get_data()
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, resolution=image_size)

            depth = depth_map.get_data()
            # with np.errstate(divide='ignore'):
            #     depth = depth.astype("uint16")
            # depth[depth == np.inf] = 0
        
            # depth = depth.astype("uint16")
            depth = depth.astype("float64")
            
            depth[depth == np.inf] = 0

            #depth = cv2.flip(depth, 1)
            self.depth_frame = depth
            # print("zed min: ", np.min(self.depth_frame), "zed max: ", np.max(self.depth_frame))
            # print("zed avg: ", np.mean(self.depth_frame), "zed median: ", np.median(self.depth_frame))
            # print("zed: ", self.depth_frame.dtype)
            self.depth_visualization_frame = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
            self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)
            self.visualize_image_frame()
            self.visualize_depth_frame()
            self.rgbd_to_point_cloud()


