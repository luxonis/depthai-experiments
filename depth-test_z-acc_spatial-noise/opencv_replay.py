import depthai as dai
import cv2
import numpy as np
import open3d as o3d
import config
from camera import Camera
import os
from pathlib import Path


class VideoReader():
    def __init__(self, left_video_path, right_video_path) -> None:
        self.left_video = cv2.VideoCapture(left_video_path)
        self.right_video = cv2.VideoCapture(right_video_path)
    def get_synced_frames(self):
        if self.left_video.isOpened() and self.right_video.isOpened():
            # print("Frame ID left", self.left_video.get(cv2.CAP_PROP_POS_FRAMES))
            # print("Frame ID right", self.right_video.get(cv2.CAP_PROP_POS_FRAMES))
            read_left, left_frame = self.left_video.read()
            read_right, right_frame = self.right_video.read()
            if not read_left or not read_right:
                self.left_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.right_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_left, left_frame = self.left_video.read()
                read_right, right_frame = self.right_video.read()
        else:
            raise RuntimeError("VideoCapture not opened")

        return left_frame, right_frame
    def get_width(self):
        return int(self.left_video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.left_video.get(cv2.CAP_PROP_FRAME_HEIGHT))



class OpenCVStereo():
    def __init__(self, calib, left_socket, right_socket, input_size) -> None:
        self.left_socket = left_socket
        self.right_socket = right_socket
        self.calib : dai.CalibrationHandler = calib
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # Hardcode the output size to input size for now
        self.output_width = input_size[0]
        self.output_height = input_size[1]
        self.alpha = config.alpha
        self.vertical = config.use_vertical
        self.set_maps()
        self.setup_stereo()

    def set_maps(self):
        leftSocket = self.left_socket
        rightSocket = self.right_socket
        calibData = self.calib

        M1 = np.array(calibData.getCameraIntrinsics(leftSocket, self.input_width, self.input_height))
        D1 = np.array(calibData.getDistortionCoefficients(leftSocket))
        M2 = np.array(calibData.getCameraIntrinsics(rightSocket, self.input_width, self.input_height))
        D2 = np.array(calibData.getDistortionCoefficients(rightSocket))

        T = np.array(calibData.getCameraTranslationVector(leftSocket, rightSocket, False))
        R = np.array(calibData.getCameraExtrinsics(leftSocket, rightSocket))[0:3, 0:3]

        modelLeft = calibData.getDistortionModel(leftSocket)
        modelRight = calibData.getDistortionModel(rightSocket)
        if modelLeft != modelRight:
            raise RuntimeError("Left and right camera models do not match")
        print("Camera model: ", modelLeft)
        model = modelLeft
        if model == dai.CameraModel.Fisheye:
            if self.vertical:
                newImageSize = (self.output_height, self.output_width)
            else:
                newImageSize = (self.output_width, self.output_height)
            R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(M1, D1, M2, D2, (self.input_width, self.input_height), R, T, balance=self.alpha, flags=cv2.fisheye.CALIB_ZERO_DISPARITY, fov_scale=1, newImageSize=newImageSize)
            if self.vertical:
                self.left_map_x, self.left_map_y = cv2.fisheye.initUndistortRectifyMap(M1, D1, R1, P1, (self.output_height, self.output_width), cv2.CV_32FC1)
                self.right_map_x, self.right_map_y = cv2.fisheye.initUndistortRectifyMap(M2, D2, R2, P2, (self.output_height, self.output_width), cv2.CV_32FC1)
            else:
                self.left_map_x, self.left_map_y = cv2.fisheye.initUndistortRectifyMap(M1, D1, R1, P1, (self.output_width, self.output_height), cv2.CV_32FC1)
                self.right_map_x, self.right_map_y = cv2.fisheye.initUndistortRectifyMap(M2, D2, R2, P2, (self.output_width, self.output_height), cv2.CV_32FC1)
        elif model == dai.CameraModel.Perspective:
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, D1, M2, D2, (self.input_width, self.input_height), R, T, alpha=self.alpha)
            self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (self.output_width, self.output_height), cv2.CV_32FC1)
            self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (self.output_width, self.output_height), cv2.CV_32FC1)
        self.focal_length_x = P1[0][0]
        self.focal_length_y = P1[1][1]
        print(P2)
        if self.vertical:
            self.baseline = abs(P2[0][3] / self.focal_length_y)
        else:
            self.baseline = abs(P2[0][3] / self.focal_length_x)
        print("Focal length x: ", self.focal_length_x)
        print("Focal length y: ", self.focal_length_y)
        print("Baseline: ", self.baseline)

    def setup_stereo(self):
        self.num_disparities = 16
        self.numDisparitySearch = 96
        blockSize = 15
        self.stereo = cv2.StereoSGBM_create( # Most similar to MX depth
            minDisparity=1,
            numDisparities=self.numDisparitySearch,
            blockSize=15,
            P1=2 * (blockSize ** 2),
            P2=3 * (blockSize ** 2),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def get_rectified(self, left, right):
        # Convert to grayscale
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        left_rect = cv2.remap(left, self.left_map_x, self.left_map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        right_rect = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

        # Pad the images
        left_rect = np.pad(left_rect, ((0,0),(self.numDisparitySearch,0)), mode='constant', constant_values=(0,0))
        right_rect = np.pad(right_rect, ((0,0),(self.numDisparitySearch,0)), mode='constant', constant_values=(0,0))

        return left_rect, right_rect

    def get_disparity(self, left_rect, right_rect):
        disparity = self.stereo.compute(left_rect, right_rect)
        return disparity

    def get_depth(self, disparity):
        depth = self.num_disparities * self.focal_length_x * self.baseline / (disparity + 1e-8)
        return depth




class OpenCVCamera(Camera):
    def __init__(self):
        super().__init__(name="OpenCV")
        if config.path is None:
            raise RuntimeError("No path to replay file provided")
        # TODO generalize based on calibration
        left_video_names = ["camb,c.avi", "left.avi", "camb,m.avi"]
        right_video_names = ["camc,c.avi", "right.avi", "camc,m.avi"]
        vertical_video_names = ["camd,c.avi", "vertical.avi", "camd,m.avi"]
        # Check if the required files exist
        video_names = [left_video_names, right_video_names, vertical_video_names] # TODO generalize based on calibration
        sockets = [dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.CAM_D]
        calib_file = Path(config.path) / Path("calib.json")
        if config.calibration_path:
            calib_file = Path(config.calibration_path)

        if not Path(config.path).exists():
            raise RuntimeError(f"Path {config.path} does not exist")

        if not calib_file.exists():
            raise RuntimeError(f"Calibration file {calib_file} not found")

        video_files = []
        for name_possibilities in video_names:
            file = None
            for name in name_possibilities:
                file = os.path.join(config.path, name)
                if os.path.exists(file):
                    break
            if file is None:
                raise RuntimeError(f"None of the files {name_possibilities} were found in {config.path}")
            video_files.append(file)

        if config.use_vertical:
            idLeft = 0
            idRight = 2
        else:
            idLeft = 0
            idRight = 1

        self.left_socket = sockets[idLeft] # TODO generalize
        self.right_socket = sockets[idRight] # TODO generalize

        self.video_reader = VideoReader(video_files[idLeft], video_files[idRight])
        self.image_size = (self.video_reader.get_width(), self.video_reader.get_height())
        self.preview_socket = sockets[idLeft]
        self.stereo = OpenCVStereo(dai.CalibrationHandler(calib_file), self.left_socket, self.right_socket, self.image_size)

        self._load_calibration(calib_file)


    def _load_calibration(self, calib_file):
        calibration = dai.CalibrationHandler(calib_file)
        self.intrinsics = calibration.getCameraIntrinsics(self.preview_socket,
            dai.Size2f(*self.image_size)
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size, self.intrinsics[0][0], self.intrinsics[1][1], self.intrinsics[0][2], self.intrinsics[1][2]
        )
        if config.use_vertical:
            self.focal_length = self.intrinsics[1][1] # in pixels
        else:
            self.focal_length = self.intrinsics[0][0] # in pixels
        self.stereoscopic_baseline = calibration.getBaselineDistance() / 100 # in m
        self.cameraModel = None
        for socket in [self.left_socket, self.right_socket]:
            self.cameraModel = calibration.getDistortionModel(socket)
            if self.cameraModel not in [dai.CameraModel.Fisheye, dai.CameraModel.Perspective]:
                raise RuntimeError(f"Unsupported camera model {self.cameraModel} - opencv stereo currently only supports Fisheye")
        self.extrinsic = np.eye(4)
        self.extrinsic[0, 3] = -0.15

    def update(self):
        self.stereo.setup_stereo()
        left_frame, right_frame = self.video_reader.get_synced_frames()
        left_rect, right_rect = self.stereo.get_rectified(left_frame, right_frame)

        disparity = self.stereo.get_disparity(left_rect, right_rect)
        # Crop all the frames for disparity
        disparity = disparity[:,self.stereo.numDisparitySearch:]
        left_rect = left_rect[:,self.stereo.numDisparitySearch:]
        right_rect = right_rect[:,self.stereo.numDisparitySearch:]

        self.depth_frame = self.stereo.get_depth(disparity) * 10 # From cm to mm
        disparityShow = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparityShow = cv2.applyColorMap(disparityShow, cv2.COLORMAP_JET)
        cv2.imshow("disparity", disparityShow)

        self.image_frame = left_rect
        self.depth_visualization_frame = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)
        cv2.imshow("depth", self.depth_visualization_frame)

        self.visualize_image_frame()
        self.rgbd_to_point_cloud()
