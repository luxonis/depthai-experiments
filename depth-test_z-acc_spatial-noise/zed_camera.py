import pyzed.sl as sl
import cv2
from camera import Camera
import time
import open3d as o3d

# Create a Camera object

class ZedCamera(Camera):
    def __init__(self):
        super().__init__(name="Zed")

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("can't open Zed camera")

        self.runtime_parameters = sl.RuntimeParameters()
        self._load_calibration()

    def __del__(self):
        self.zed.close()


    def _load_calibration(self):
        intrinsic_parameters = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.zed.get_camera_information().camera_configuration.resolution.width,
            self.zed.get_camera_information().camera_configuration.resolution.height,
            intrinsic_parameters.fx, intrinsic_parameters.fy, intrinsic_parameters.cx, intrinsic_parameters.cy
        )
    def update(self):
        image_size = sl.Resolution(int(self.zed.get_camera_information().camera_configuration.resolution.width ),
                                   int(self.zed.get_camera_information().camera_configuration.resolution.height ))
        left_image = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        depth_map = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT, resolution=image_size)
            self.image_frame  = left_image.get_data()
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, resolution=image_size)

            depth_image_ocv = depth_map.get_data()
            depth_image_ocv = depth_image_ocv.astype("uint16")
            depth = cv2.flip(depth_image_ocv, 1)
            self.depth_frame = depth

            self.depth_visualization_frame = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
            self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)
            cv2.imshow("Depth", self.depth_visualization_frame)
            self.visualize_image_frame()
            self.rgbd_to_point_cloud()
