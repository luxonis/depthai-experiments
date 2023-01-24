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
        super().__init__(name="Replay")

        self.device_info = device_info
        self.mxid = device_info.getMxId()
        self.use_replay = False
        self.stop = False
        self.frames = []
        self.idx = 0
        self.depth_frames = []
        self.removed_frames = False

        if os.path.exists(config.path + '/cama,c.avi'):
            os.rename(config.path + '/cama,c.avi', config.path + '/rgb.avi')
            os.rename(config.path + '/camb,c.avi', config.path + '/left.avi')
            os.rename(config.path + '/camc,c.avi', config.path + '/right.avi')
       
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


    def getMesh(self, calibData, resolution, offset, rectificationScale):
        print("------mesh res", resolution, "offset", offset) # TODO see if offset is needed here and implement...
        width, height = resolution
        offsetWidth, offsetHeight = offset
        ## Top left and bottom right are from camera perspective where Top left corner is at (0,0) and bottom right is at (width, height)
        topLeftPixel = dai.Point2f(offsetWidth, offsetHeight) 
        bottomRightPixel = dai.Point2f(resolution[0] + offsetWidth , resolution[1] + offsetHeight)

        print(topLeftPixel.x, topLeftPixel.y)
        print(bottomRightPixel.x, bottomRightPixel.y)
        def_m1, w, h = calibData.getDefaultIntrinsics(calibData.getStereoLeftCameraId())
        """ def_m1 = np.array(def_m1) * 1.5
        def_m1[0,2] -= (1920-1280) / 2 """

        print("Def Intrins---")
        print(w,h)
        print(np.array(def_m1))
        # we are using original height and width since we don't want to resize. But just crop based on the corner pixel positions
        M1 = np.array(calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId(), w, h, topLeftPixel, bottomRightPixel))
        M2 = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), w, h, topLeftPixel, bottomRightPixel))
        print("cropped Intrins")
        print(resolution)
        print(M1)
        
        d1 = np.array(calibData.getDistortionCoefficients(calibData.getStereoLeftCameraId()))
        d2 = np.array(calibData.getDistortionCoefficients(calibData.getStereoRightCameraId()))

        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())

        tranformation = np.array(calibData.getCameraExtrinsics(calibData.getStereoLeftCameraId(), calibData.getStereoRightCameraId()))
        R = tranformation[:3, :3]
        T = tranformation[:3, 3]

        rectIntrinsics = M2.copy()

        if rectificationScale > 0 and rectificationScale < 1:
            rectIntrinsics[0][0] *= rectificationScale
            rectIntrinsics[1][1] *= rectificationScale

        mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, rectIntrinsics, resolution, cv2.CV_32FC1)
        mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, rectIntrinsics, resolution, cv2.CV_32FC1)

        meshCellSize = 16
        meshLeft = []
        meshRight = []

        for y in range(mapXL.shape[0] + 1):
            if y % meshCellSize == 0:
                rowLeft = []
                rowRight = []
                for x in range(mapXL.shape[1] + 1):
                    if x % meshCellSize == 0:
                        if y == mapXL.shape[0] and x == mapXL.shape[1]:
                            rowLeft.append(mapYL[y - 1, x - 1])
                            rowLeft.append(mapXL[y - 1, x - 1])
                            rowRight.append(mapYR[y - 1, x - 1])
                            rowRight.append(mapXR[y - 1, x - 1])
                        elif y == mapXL.shape[0]:
                            rowLeft.append(mapYL[y - 1, x])
                            rowLeft.append(mapXL[y - 1, x])
                            rowRight.append(mapYR[y - 1, x])
                            rowRight.append(mapXR[y - 1, x])
                        elif x == mapXL.shape[1]:
                            rowLeft.append(mapYL[y, x - 1])
                            rowLeft.append(mapXL[y, x - 1])
                            rowRight.append(mapYR[y, x - 1])
                            rowRight.append(mapXR[y, x - 1])
                        else:
                            rowLeft.append(mapYL[y, x])
                            rowLeft.append(mapXL[y, x])
                            rowRight.append(mapYR[y, x])
                            rowRight.append(mapXR[y, x])
                if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                    rowLeft.append(0)
                    rowLeft.append(0)
                    rowRight.append(0)
                    rowRight.append(0)

                meshLeft.append(rowLeft)
                meshRight.append(rowRight)

        meshLeft = np.array(meshLeft)
        meshRight = np.array(meshRight)

        meshLeft = list(meshLeft.tobytes())
        meshRight = list(meshRight.tobytes())

        return meshLeft, meshRight, rectIntrinsics

    def _create_pipeline(self):
        pipeline, nodes = self.replay.init_pipeline()

        calibData = self.replay.calibData
        self.image_size = self.replay.get_size(None)

        originalRes = (1920, 1200)
        res = (1280, 1080)
    
        centerCropOffset = ((originalRes[0] - res[0]) / 2, (originalRes[1] - res[1]) / 2)

        leftMesh, rightMesh, self.M2 = self.getMesh(calibData, res, centerCropOffset, 0.5)

        nodes.stereo.loadMeshData(leftMesh, rightMesh)
        nodes.stereo.setSubpixel(True)

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

            if len(self.frames) > 6 and not self.removed_frames:
                self.remove_frames = True
                self.frames = []
                self.depth_frames = []

        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)

        cv2.imshow('depth', self.depth_visualization_frame)
        
        self.visualize_image_frame()
        self.rgbd_to_point_cloud()
