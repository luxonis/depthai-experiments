import depthai as dai
import cv2
import numpy as np
import open3d as o3d
from typing import List
import config
from camera import Camera
from host_sync import HostSync
from stereo_rvc3 import pipeline_creation
from stereo_config import StereoConfigHandler

class OakCamera(Camera):
    def __init__(self, device_info: dai.DeviceInfo, vertical=True, use_opencv=False):
        super().__init__(name="OAK")
        self.use_opencv = use_opencv
        self.vertical = vertical
        self.device_info = device_info
        self.device = dai.Device() # TODO add device_info
        self._create_pipeline()
        self._load_calibration()
        self.device.startPipeline(self.pipeline)
        self.mxid = self.device.getDeviceInfo().getMxId()
        if self.use_opencv:
            stereo = cv2.StereoSGBM_create()
            # Set the parameters for StereoSGBM
            stereo.setBlockSize(9)
            stereo.setMinDisparity(0)
            stereo.setNumDisparities(64)
            stereo.setUniquenessRatio(10)
            stereo.setSpeckleWindowSize(0)
            stereo.setSpeckleRange(0)
            stereo.setDisp12MaxDiff(0)
            self.opencv_stereo = stereo

        self.image_queue = self.device.getOutputQueue(name=self.image_name, maxSize=10, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name=self.depth_name, maxSize=10, blocking=False)
        self.right_image_queue = self.device.getOutputQueue(name=self.right_image_name, maxSize=10, blocking=False)
        self.host_sync = HostSync([self.image_name, self.depth_name, self.right_image_name])
        self.input_config_queue = self.device.getInputQueue(self.input_config_name)


    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())

    def _load_calibration(self):
        if config.args.calib is None:
            self.calibration = self.device.readCalibration()
        else:
            self.calibration = dai.CalibrationHandler(config.args.calib)

        self.intrinsics = self.calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.LEFT,
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )

        self.focal_length = self.intrinsics[0][0] # in pixels
        self.stereoscopic_baseline = self.calibration.getBaselineDistance()/100 # in m

        self.extrinsic = np.eye(4)
        self.extrinsic[0, 3] = -0.15


    def _create_pipeline(self):
        # Create pipeline
        pipeline = dai.Pipeline()
        if config.args.calib is None:
            self.calibration = self.device.readCalibration()
        else:
            self.calibration = dai.CalibrationHandler(config.args.calib)
        calibData = self.calibration
        # Cameras
        colorLeft = pipeline.create(dai.node.ColorCamera)
        colorRight = pipeline.create(dai.node.ColorCamera)
        colorVertical = pipeline.create(dai.node.ColorCamera)
        colorVertical.setBoardSocket(dai.CameraBoardSocket.CAM_D)
        colorVertical.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        colorLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        colorLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        colorRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        colorRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        


        name = "horizontal"
        if self.vertical:
            #################################################  vertical ###############################################################
            name = "vertical"
            stereo, outNames = pipeline_creation.create_stereo(pipeline, name, colorLeft.isp, colorVertical.isp, False, True, True)
            self.image_size = (colorLeft.getResolutionHeight(), colorLeft.getResolutionWidth())
            self.mono_image_size = self.image_size

            meshLeft, meshVertical, self.scale_factor = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorVertical.getBoardSocket(),
                                                                        (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight()), vertical=True)
            stereo.loadMeshData(meshLeft, meshVertical)
            stereo.setVerticalStereo(True)
            stereo.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
            ############################################################################################################################

        else:
            #################################################  horizontal ##############################################################
            name = "horizontal"
            stereo, outNames = pipeline_creation.create_stereo(pipeline, name, colorLeft.isp, colorRight.isp, False, True, True)
            self.image_size = (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight())
            self.mono_image_size = self.image_size
            meshLeft, meshVertical, self.scale_factor = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorRight.getBoardSocket(),
                                                                        (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight()))
            stereo.loadMeshData(meshLeft, meshVertical)
            stereo.setLeftRightCheck(False)
            stereo.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
            ############################################################################################################################
        if self.use_opencv:
            self.scale_factor /= 2

        StereoConfigHandler(stereo.initialConfig.get())
        StereoConfigHandler.registerWindow(name + ' stereo control panel')

        xinStereoDepthConfig = pipeline.create(dai.node.XLinkIn)
        xinStereoDepthConfigName = name + "-stereoDepthConfig"
        xinStereoDepthConfig.setStreamName(xinStereoDepthConfigName)
        xinStereoDepthConfig.out.link(stereo.inputConfig)

        rectified_left, rectified_right, disparity = outNames
        self.depth_name = disparity
        self.image_name = rectified_left
        self.right_image_name = rectified_right
        self.input_config_name = xinStereoDepthConfigName

        self.pipeline = pipeline

    def update(self, key = None):
        
        if key is not None:
            StereoConfigHandler.handleKeypress(key, self.input_config_queue)


        for queue in [self.depth_queue, self.image_queue, self.right_image_queue]:
            new_msgs = queue.tryGetAll()
            if new_msgs is not None:
                for new_msg in new_msgs:
                    new_msg.setSequenceNum(0) # Dirty hack to make it work, as sequence numbers are not correct on RVC3 
                    self.host_sync.add(queue.getName(), new_msg)

        msg_sync = self.host_sync.get()
        if msg_sync is None:
            return
        self.image_frame = msg_sync[self.image_name].getCvFrame()
        self.right_image_frame = msg_sync[self.right_image_name].getCvFrame()
        self.depth_frame = msg_sync[self.depth_name].getFrame()

        if self.use_opencv:
            self.depth_frame = self.opencv_stereo.compute(self.image_frame, self.right_image_frame)

        with np.errstate(divide='ignore'):
            self.depth_frame = self.scale_factor / self.depth_frame
        self.depth_frame[self.depth_frame == np.inf] = 0

        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)
        self.visualize_image_frame()
        self.visualize_depth_frame()
        self.rgbd_to_point_cloud()