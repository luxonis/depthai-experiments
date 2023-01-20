import depthai as dai
import cv2
import numpy as np
import time
import config
import os

class Camera:
    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 360)

        # Camera intrinsic parameters
        self.intrinsic_mat = np.array(self.device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.RGB, 3840, 2160))
        self.distortion_coef = np.zeros((1,5))

        # Camera extrinsic parameters
        self.rot_vec = None
        self.trans_vec = None
        self.world_to_cam = None
        self.cam_to_world = None

        self.checkerboard_size = config.checkerboard_size
        self.checkerboard_inner_size = (self.checkerboard_size[0] - 1, self.checkerboard_size[1] - 1)
        self.square_size = config.square_size
        self.corners_world = np.zeros((1, self.checkerboard_inner_size[0] * self.checkerboard_inner_size[1], 3), np.float32)
        self.corners_world[0,:,:2] = np.mgrid[0:self.checkerboard_inner_size[0], 0:self.checkerboard_inner_size[1]].T.reshape(-1, 2)
        self.corners_world *= self.square_size

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())
    
    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(640, 360)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName('control')
        control.out.link(cam_rgb.inputControl)

        self.pipeline = pipeline

    def update(self):
        in_rgb = self.rgb_queue.tryGet()

        if in_rgb is None:
            return
        
        self.frame_rgb = in_rgb.getCvFrame()

        cv2.imshow(self.window_name, self.frame_rgb)

    def capture_still(self, timeout_ms: int = 1000):
        print("capturing still")
        # Empty the queue
        self.still_queue.tryGetAll()

        # Send a capture command
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still = None
        start_time = time.time()*1000
        while in_still is None:
            time.sleep(0.1)
            in_still = self.still_queue.tryGet()
            if time.time()*1000 - start_time > timeout_ms:
                print("did not recieve still image - retrying")
                return self.capture_still(timeout_ms)

        still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)

        return still_rgb

    def draw_origin(self, frame_rgb: np.ndarray):
        points, _ = cv2.projectPoints(
            np.float64([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]), 
            self.rot_vec, self.trans_vec, self.intrinsic_mat, self.distortion_coef
        )
        [p_0, p_x, p_y, p_z] = points.astype(np.int64)

        reprojection = frame_rgb.copy()
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_x[0]), (0, 0, 255), 5)
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_y[0]), (0, 255, 0), 5)
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_z[0]), (255, 0, 0), 5)

        return reprojection

    def estimate_pose(self):
        frame_rgb = self.capture_still()
        if frame_rgb is None:
            print("did not recieve still image")
            return

        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        
        print("Finding checkerboard corners...")

        # find the checkerboard corners
        found, corners = cv2.findChessboardCorners(
            frame_gray, self.checkerboard_inner_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print("Checkerboard not found")
            return None

        # refine the corner locations
        corners = cv2.cornerSubPix(
            frame_gray, corners, (11, 11), (-1, -1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # compute the rotation and translation from the camera to the checkerboard
        ret, self.rot_vec, self.trans_vec = cv2.solvePnP(
            self.corners_world, corners, self.intrinsic_mat, self.distortion_coef
        )

        # compute transformation from world to camera space and wise versa
        rotM = cv2.Rodrigues(self.rot_vec)[0]
        self.world_to_cam = np.vstack((np.hstack((rotM, self.trans_vec)), np.array([0,0,0,1])))
        self.cam_to_world = np.linalg.inv(self.world_to_cam)

        # show origin overlay
        reprojection = self.draw_origin(frame_rgb)
        cv2.imshow(self.window_name, reprojection)
        cv2.waitKey()

        print("Camera to world transformation: \n", self.cam_to_world)
        print("World to camera transformation: \n", self.world_to_cam)
        print("Rotation vector: \n", self.rot_vec)
        print("Translation vector: \n", self.trans_vec)

        # save the results
        try:
            path = os.path.join(os.path.dirname(__file__), f"{config.calibration_data_dir}")
            os.makedirs(path, exist_ok=True)
            np.savez(
                f"{path}/extrinsics_{self.device_info.getMxId()}.npz", 
                world_to_cam=self.world_to_cam, cam_to_world=self.cam_to_world, trans_vec=self.trans_vec, rot_vec=self.rot_vec
            )
        except:
            print("Could not save calibration data")