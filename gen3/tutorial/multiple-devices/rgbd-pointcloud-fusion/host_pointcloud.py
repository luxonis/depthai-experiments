from threaded_display import PointCloudVisualizer
from typing import Tuple
import config
import numpy as np
import depthai as dai
import open3d as o3d
import cv2


class PointCloud(dai.node.HostNode):
    def __init__(self, callback_frame : callable, window_name : str, 
                 device : dai.Device, image_size : Tuple[int, int], pcl_converter : PointCloudVisualizer) -> None:
        super().__init__()
        self.callback_frame = callback_frame
        self.window_name = window_name
        self.device = device
        self.image_size = image_size

        self.point_cloud_alignment = None
        self.pinhole_camera_intrinsic = None
        self.intrinsics = None
        self.cam_to_world = None
        self.world_to_cam = None
        self.point_cloud = None

        self.dx_id = device.getMxId()

        self.pcl_converter = pcl_converter
        


    def build(self, depth_out : dai.Node.Output, cam_isp : dai.Node.Output) -> "PointCloud":
        self.link_args(depth_out, cam_isp)
        self.sendProcessingToPipeline(True)

        self._load_calibration()
        self.pcl_converter.set_params(self.pinhole_camera_intrinsic, self.cam_to_world, self.point_cloud_alignment)

        return self


    def process(self, depth_frame : dai.ImgFrame, cam_isp : dai.ImgFrame) -> None:
        depth_frame = depth_frame.getFrame()
        rgb_frame = cam_isp.getCvFrame()

        depth_visualization_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_visualization_frame = cv2.equalizeHist(depth_visualization_frame)
        depth_visualization_frame = cv2.applyColorMap(depth_visualization_frame, cv2.COLORMAP_HOT)

        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        self.point_cloud = self.pcl_converter.rgbd_to_point_cloud(depth_frame, rgb)
        
        self.callback_frame(self.point_cloud, self.window_name)


    def _load_calibration(self):
        path = f"{config.calibration_data_dir}"
        try:
            extrinsics = np.load(f"{path}/extrinsics_{self.dx_id}.npz")
            self.cam_to_world = extrinsics["cam_to_world"]
            self.world_to_cam = extrinsics["world_to_cam"]
        except:
            raise RuntimeError(f"Could not load calibration data for camera {self.dx_id} from {path}!")

        calibration = self.device.readCalibration()
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A if config.COLOR else dai.CameraBoardSocket.CAM_C, 
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )

        try:
            self.point_cloud_alignment = np.load(f"{config.calibration_data_dir}/point_cloud_alignment_{self.dx_id}.npy")
        except:
            self.point_cloud_alignment = np.eye(4)

        print(self.pinhole_camera_intrinsic)
