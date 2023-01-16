import depthai as dai
import cv2
import numpy as np
import open3d as o3d
from typing import List
import config
from camera import Camera
from host_sync import HostSync
import os
from depthai_replay import Replay

class ReplayCamera(Camera):
    def __init__(self, device_info: dai.DeviceInfo):
        super().__init__(name="OAK")

        self.device_info = device_info
        self.mxid = device_info.getMxId()
        self.use_replay = False
        self.stop = False
        self.frames = []
        self.idx = 0
        self.depth_frames = []


        # os.rename(config.path + '/cama,c.avi', config.path + '/rgb.avi')
        # os.rename(config.path + '/camb,c.avi', config.path + '/left.avi')
        # os.rename(config.path + '/camc,c.avi', config.path + '/right.avi')
        self.replay = Replay(config.path)
       
        self._create_pipeline()

        self.device = dai.Device(self.pipeline, self.device_info)
        self.replay.create_queues(self.device)

        self.image_queue = self.device.getOutputQueue(name="image", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.host_sync = HostSync(["image", "depth"])
        
        self._load_calibration()

    def __del__(self):
        if self.use_replay:
            self.replay.close()
            print("=== Closed replay")

        self.device.close()
        print("=== Closed ") #+ self.device_info.getMxId())

    def _load_calibration(self):
        calibration =  self.replay.calibData
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB if config.COLOR else dai.CameraBoardSocket.RIGHT, 
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )

        self.focal_length = self.intrinsics[0][0] # in pixels
        self.stereoscopic_baseline = calibration.getBaselineDistance()/100 # in m


    def get_mesh(self, mapX, mapY):
        meshCellSize = 16
        mesh0 = []
        print(mapX.shape)
        # Creates subsampled mesh which will be loaded on to device to undistort the image
        for y in range(mapX.shape[0] + 1):  # iterating over height of the image
            if y % meshCellSize == 0:
                rowLeft = []
                for x in range(mapX.shape[1]+ 1):  # iterating over width of the image
                    if x % meshCellSize == 0:
                        if y == mapX.shape[0] and x == mapX.shape[1]:
                            rowLeft.append(mapY[y - 1, x - 1])
                            rowLeft.append(mapX[y - 1, x - 1])
                        elif y == mapX.shape[0]:
                            rowLeft.append(mapY[y - 1, x])
                            rowLeft.append(mapX[y - 1, x])
                        elif x == mapX.shape[1]:
                            rowLeft.append(mapY[y, x - 1])
                            rowLeft.append(mapX[y, x - 1])
                        else:
                            rowLeft.append(mapY[y, x])
                            rowLeft.append(mapX[y, x])
                if (mapX.shape[1] % meshCellSize) % 2 != 0:
                    rowLeft.append(0)
                    rowLeft.append(0)

                mesh0.append(rowLeft)
        mesh0 = np.array(mesh0)

        return mesh0

    def get_maps(self, width, height, calib):
        imageSize = (width, height)
        print(f'Image size is {imageSize}')
        M1 = np.array(calib.getCameraIntrinsics(calib.getStereoLeftCameraId(), width, height))
        M2 = np.array(calib.getCameraIntrinsics(calib.getStereoRightCameraId(), width, height))
        d1 = np.array(calib.getDistortionCoefficients(calib.getStereoLeftCameraId()))
        d2 = np.array(calib.getDistortionCoefficients(calib.getStereoRightCameraId()))
        R1 = np.array(calib.getStereoLeftRectificationRotation())
        R2 = np.array(calib.getStereoRightRectificationRotation())
        
        increaseOffset = 0
        M2_focal = M2.copy()
        M2_focal[0][0] += increaseOffset
        M2_focal[1][1] += increaseOffset
        kScaledL = M2_focal
        kScaledR = kScaledL
        
        
        M11, a = cv2.getOptimalNewCameraMatrix(M1, d1, (width, height), alpha=0)

        fov_x = np.rad2deg(2 * np.arctan2(a[2], 2 * M11[0][0]))
        fov_y = np.rad2deg(2 * np.arctan2(a[3], 2 * M11[1][1]))

        print("Field of View (degrees):")
        print(f"  {fov_x = :.1f}\N{DEGREE SIGN}")
        print(f"  {fov_y = :.1f}\N{DEGREE SIGN}")

        mapX_l, mapY_l = cv2.initUndistortRectifyMap(M1, d1, R1, kScaledL, imageSize, cv2.CV_32FC1)
        mapX_r, mapY_r = cv2.initUndistortRectifyMap(M2, d2, R2, kScaledR, imageSize, cv2.CV_32FC1)

        return mapX_l, mapY_l, mapX_r, mapY_r, M2    

    def _create_pipeline(self):
        pipeline, nodes = self.replay.init_pipeline()

        calibData = self.replay.calibData
        self.image_size = self.replay.get_size(None)
        print(self.image_size)

        mapX_left, mapY_left, mapX_right, mapY_right, self.M2 = self.get_maps(self.image_size[0], self.image_size[1], calibData)
        
        mesh_l = self.get_mesh(mapX_left, mapY_left)
        mesh_r = self.get_mesh(mapX_right, mapY_right)
        meshLeft = list(mesh_l.tobytes())
        meshRight = list(mesh_r.tobytes())
        nodes.stereo.loadMeshData(meshLeft, meshRight)
        nodes.stereo.setSubpixel(True)
        #nodes.stereo.setExtendedDisparity(True)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        nodes.stereo.disparity.link(xout_depth.input)
        
        # RGB cam or mono right -> 'image'
        xout_image = pipeline.createXLinkOut()
        xout_image.setStreamName("image")

        nodes.stereo.rectifiedRight.link(xout_image.input)

        self.mono_image_size = self.replay.get_size(None)
        self.pipeline = pipeline
        self.nodes = nodes


    def update(self):

        if not self.replay.send_frames(False):
            self.stop = True
            self.depth_frame = self.depth_frames[self.idx % len(self.depth_frames)]
            self.image_frame = self.frames[self.idx % len(self.frames)]
            self.idx = self.idx + 1

        else:
            
            disparity = self.depth_queue.get().getFrame()

            with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
                self.depth_frame = (8 * self.M2[0][0] * self.replay.calibData.getBaselineDistance() * 10 / disparity).astype(np.uint16)

            self.image_frame = self.image_queue.get().getCvFrame()

            self.frames.append(self.image_frame)
            self.depth_frames.append(self.depth_frame)

        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)

        cv2.imshow('depth', self.depth_visualization_frame)
        

        self.visualize_image_frame()
        self.rgbd_to_point_cloud()