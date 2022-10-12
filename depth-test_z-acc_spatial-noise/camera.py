import depthai as dai
import cv2
import numpy as np
import open3d as o3d
from typing import List
import config
from host_sync import HostSync

class Camera:
    def __init__(self, device_info: dai.DeviceInfo):
        self.device_info = device_info
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.device.setIrLaserDotProjectorBrightness(1200)

        self.image_queue = self.device.getOutputQueue(name="image", maxSize=10, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=10, blocking=False)
        self.host_sync = HostSync(["image", "depth"])

        self.image_frame = None
        self.depth_frame = None
        self.depth_visualization_frame = None
        self.point_cloud = o3d.geometry.PointCloud()

        self._load_calibration()

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())

    def _load_calibration(self):
        calibration = self.device.readCalibration()
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB if config.COLOR else dai.CameraBoardSocket.RIGHT, 
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )
    

    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # Depth cam -> 'depth'
        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.createStereoDepth()
        cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.initialConfig.setMedianFilter(config.median)
        cam_stereo.initialConfig.setConfidenceThreshold(config.confidence_threshold)
        cam_stereo.setLeftRightCheck(config.lrcheck)
        cam_stereo.setExtendedDisparity(config.extended)
        cam_stereo.setSubpixel(config.subpixel)
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        init_config = cam_stereo.initialConfig.get()
        init_config.postProcessing.speckleFilter.enable = False
        init_config.postProcessing.speckleFilter.speckleRange = 50
        init_config.postProcessing.temporalFilter.enable = True
        init_config.postProcessing.spatialFilter.enable = True
        init_config.postProcessing.spatialFilter.holeFillingRadius = 2
        init_config.postProcessing.spatialFilter.numIterations = 1
        init_config.postProcessing.thresholdFilter.minRange = config.min_range
        init_config.postProcessing.thresholdFilter.maxRange = config.max_range
        init_config.postProcessing.decimationFilter.decimationFactor = 1
        cam_stereo.initialConfig.set(init_config)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        cam_stereo.depth.link(xout_depth.input)

        
        # RGB cam or mono right -> 'image'
        xout_image = pipeline.createXLinkOut()
        xout_image.setStreamName("image")
        if config.COLOR:
            cam_rgb = pipeline.createColorCamera()
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setIspScale(1, 3)
            cam_rgb.initialControl.setManualFocus(130)
            cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            cam_rgb.isp.link(xout_image.input)
            self.image_size = cam_rgb.getIspSize()
        else:
            cam_stereo.rectifiedRight.link(xout_image.input)
            self.image_size = mono_right.getResolutionSize()

        self.pipeline = pipeline

    def update(self):
        for queue in [self.depth_queue, self.image_queue]:
            new_msgs = queue.tryGetAll()
            if new_msgs is not None:
                for new_msg in new_msgs:
                    self.host_sync.add(queue.getName(), new_msg)

        msg_sync = self.host_sync.get()
        if msg_sync is None:
            return
        
        self.depth_frame = msg_sync["depth"].getFrame()
        self.image_frame = msg_sync["image"].getCvFrame()
        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)


    def rgbd_to_point_cloud(self, depth_frame, image_frame, downsample=False, remove_noise=False):
        rgb_o3d = o3d.geometry.Image(image_frame)
        df = np.copy(depth_frame).astype(np.float32)
        # df -= 20
        depth_o3d = o3d.geometry.Image(df)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(image_frame.shape) != 3)
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic
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