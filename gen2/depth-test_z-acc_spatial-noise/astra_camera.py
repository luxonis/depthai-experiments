from turtle import width
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import open3d as o3d
from camera import Camera
import config

class AstraCamera(Camera):
    def __init__(self, device: openni2.Device):
        super().__init__(name="Astra")

        self.device = device
        self.width = 640
        self.height = 480

        self.image_stream = device.create_color_stream()
        self.image_stream.start()
        self.image_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, 
            resolutionX=self.width, resolutionY=self.height, fps=30
        ))

        self.depth_stream = device.create_depth_stream()
        self.depth_stream.start()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
            resolutionX=self.width, resolutionY=self.height, fps = 30
        ))

        self._load_calibration()

    def __del__(self):
        self.image_stream.stop()
        self.depth_stream.stop()

    def _load_calibration(self):
        # Intrinsics (https://3dclub.orbbec3d.com/t/access-intrinsic-camera-parameters/2784)
        hfov = self.image_stream.get_horizontal_fov()
        vfov = self.image_stream.get_vertical_fov()

        try: # try to load intrinsic parameters from file (specified with --astra_intrinsic)
            intrinsics = np.load(config.astra_intrinsic)
            self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.width, self.height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
            )
            print("Astra camera intrinsics: ", intrinsics)
        except:
            self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.width,
                height=self.height,
                fx = self.width / (2 * np.tan(hfov / 2)),
                fy = self.height / (2 * np.tan(vfov / 2)),
                cx = self.width / 2,
                cy = self.height / 2
            )



    def update(self):
        # Get the rgb frame
        bgr_frame = np.fromstring(self.image_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(self.height, self.width, 3)
        bgr_frame = cv2.flip(bgr_frame, 1)
        self.image_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Grab a new depth frame
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        # Put the depth frame into a numpy array and reshape it
        depth: np.ndarray = np.frombuffer(frame_data, dtype=np.uint16).reshape(self.height, self.width)
        depth = cv2.flip(depth, 1)
        self.depth_frame = depth

        self.depth_visualization_frame = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        self.depth_visualization_frame = cv2.equalizeHist(self.depth_visualization_frame)
        self.depth_visualization_frame = cv2.applyColorMap(self.depth_visualization_frame, cv2.COLORMAP_HOT)

        self.visualize_image_frame()
        self.rgbd_to_point_cloud()