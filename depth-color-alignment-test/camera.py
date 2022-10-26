import depthai as dai
import cv2
import numpy as np
from typing import List
from host_sync import HostSync
import config

class Camera:
    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True):
        self.show_video = show_video
        self.show_depth = False
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.device.setIrLaserDotProjectorBrightness(1200)

        self.image_queue = self.device.getOutputQueue(name="image", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.host_sync = HostSync(["image", "depth"])

        self.image_frame = None
        self.depth_frame = None
        self.depth_visualization_frame = None

        # camera window
        self.image_window_name = f"[{self.friendly_id}] Camera RGB - mxid: {self.mxid}"
        self.depth_window_name = f"[{self.friendly_id}] Camera depth - mxid: {self.mxid}"
        if show_video:
            cv2.namedWindow(self.image_window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.image_window_name, 640, 360)
            cv2.resizeWindow(self.depth_window_name, 640, 360)

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

    def save_point_cloud_alignment(self):
        np.save(f"{config.calibration_data_dir}/point_cloud_alignment_{self.mxid}.npy", self.point_cloud_alignment)
    

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
        self.depth_visualization_frame[self.depth_frame <= 1] = np.array([255, 255, 0])

        if self.show_video:
            cv2.imshow(self.depth_window_name, self.depth_visualization_frame)
            cv2.imshow(self.image_window_name, self.image_frame)