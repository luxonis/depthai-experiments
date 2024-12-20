import cv2
import depthai as dai
import threading
import os
import config
import numpy as np
import time
from utility import filter_internal_cameras, run_pipeline

class OpencvManager:
    def __init__(self):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.keys = []
        self.frames : dict[str, np.ndarray] = {}
        self.ctrl_queues : dict[str, dai.InputQueue] = {}
        self.cam_stills : dict[str, dai.MessageQueue] = {}
        self.intrinsic_mats : dict[int, np.ndarray] = {}
        self.cameras : dict[int, str] = {}
        self.dx_ids : dict[int, str] = {}
        self.selected_camera = None

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

        self.distortion_coef = np.zeros((1,5))


    def run(self) -> None:
        for window_name in self.frames.keys():
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 360)

        while True:
           
            self.newFrameEvent.wait()
            for name in self.frames.keys():
                if self.frames[name] is not None:
                    key = cv2.waitKey(1)

                    # QUIT - press `q` to quit
                    if key == ord('q'):
                        for dx_id in self.dx_ids.values():
                            print("=== Closed " + dx_id)
                        return
                    
                    # CAMERA SELECTION - use the number keys to select a camera
                    if key >= ord('1') and key <= ord('9'):
                        self._select_camera(key - ord('1') + 1)

                    # POSE ESTIMATION - press `p` to estimate the pose of the selected camera and save it to file
                    if key == ord('p'):
                        self._estimate_pose()

                    cv2.imshow(name, self.frames[name])
                    

    def set_frame(self, frame : dai.ImgFrame, window_name : str) -> None:
        with self.lock:
            self.frames[window_name] = frame
            self.newFrameEvent.set()

    
    def set_params(self, window_name : str, ctrl_queue : dai.InputQueue, cam_still : dai.MessageQueue, friendly_id : int, device : dai.Device) -> None:
        self.ctrl_queues[window_name] = ctrl_queue
        self.cam_stills[window_name] = cam_still
        self.cameras[friendly_id] = window_name
        self.dx_ids[friendly_id] = device.getMxId()
        self.intrinsic_mats[friendly_id] = np.array(device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 3840, 2160))
        self.selected_camera = 1


    def set_custom_key(self, key : str) -> None:
        self.keys.append(key)
        self._init_frames()

    
    def _init_frames(self) -> None:
        for key in self.keys:
            if key not in self.frames.keys():
                self.frames[key] = None
    

    def _select_camera(self, friendly_id: int) -> None:
        i = friendly_id - 1
        if i >= len(self.frames) or i < 0: 
            return None 
        
        self.selected_camera = i
        print(f"Selected camera {friendly_id}")


    def _capture_still(self, timeout_ms: int = 1000):
        print("capturing still")
        # Empty the queue
        camera_name = self.cameras[self.selected_camera]
        still_queue = self.cam_stills[camera_name]
        still_queue.tryGetAll()

        # Send a capture command
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        control_queue = self.ctrl_queues[camera_name]
        control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still : dai.ImgFrame | None = None
        start_time = time.time()*1000
        while in_still is None:
            time.sleep(0.1)
            in_still = still_queue.tryGet()
            if time.time()*1000 - start_time > timeout_ms:
                print("did not recieve still image - retrying")
                return self._capture_still(timeout_ms)

        still_rgb = cv2.imdecode(in_still.getFrame(), cv2.IMREAD_UNCHANGED)

        return still_rgb


    def _draw_origin(self, frame_rgb: np.ndarray) -> np.ndarray:
        points, _ = cv2.projectPoints(
            np.float64([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]), 
            self.rot_vec, self.trans_vec, self.intrinsic_mats[self.selected_camera], self.distortion_coef
        )
        [p_0, p_x, p_y, p_z] = points.astype(np.int64)

        reprojection = frame_rgb.copy()
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_x[0]), (0, 0, 255), 5)
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_y[0]), (0, 255, 0), 5)
        reprojection = cv2.line(reprojection, tuple(p_0[0]), tuple(p_z[0]), (255, 0, 0), 5)

        return reprojection

    
    def _estimate_pose(self) -> None:
        frame_rgb = self._capture_still()
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
            self.corners_world, corners, self.intrinsic_mats[self.selected_camera], self.distortion_coef
        )

        # compute transformation from world to camera space and wise versa
        rotM = cv2.Rodrigues(self.rot_vec)[0]
        self.world_to_cam = np.vstack((np.hstack((rotM, self.trans_vec)), np.array([0,0,0,1])))
        self.cam_to_world = np.linalg.inv(self.world_to_cam)

        # show origin overlay
        reprojection = self._draw_origin(frame_rgb)
        cv2.imshow(self.cameras[self.selected_camera], reprojection)
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
                f"{path}/extrinsics_{self.dx_ids[self.selected_camera]}.npz", 
                world_to_cam=self.world_to_cam, cam_to_world=self.cam_to_world, trans_vec=self.trans_vec, rot_vec=self.rot_vec
            )
        except:
            print("Could not save calibration data")



class Display(dai.node.HostNode):
    def __init__(self, callback_frame : callable, callback_params : callable, window_name : str, device : dai.Device, friendly_id : int) -> None:
        super().__init__()
        self.callback_frame = callback_frame
        self.callback_params = callback_params
        self.window_name = window_name
        self.device = device
        self.friendly_id = friendly_id

    
    def build(self, cam_preview : dai.Node.Output, cam_still : dai.Node.Output, ctrl_queue : dai.InputQueue) -> "Display":  
        self.inputs["in_frame"].setMaxSize(1)
        self.inputs["in_frame"].setBlocking(False)
        self.ctrl_queue = ctrl_queue
        self.cam_still_queue = cam_still.createOutputQueue(maxSize=1, blocking=False)
        
        self.callback_params(self.window_name, self.ctrl_queue, self.cam_still_queue, self.friendly_id, self.device)

        self.link_args(cam_preview)
        self.sendProcessingToPipeline(True)
        return self 
    

    def process(self, in_frame : dai.ImgFrame) -> None:
        self.callback_frame(in_frame.getCvFrame(), self.window_name)


def get_pipelines(device : dai.Device, callback_frame : callable, callback_params : callable, friendly_id : int) -> dai.Pipeline:
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_preview = cam_rgb.requestOutput(size=(640, 360), type=dai.ImgFrame.Type.NV12)

    still_encoder = pipeline.create(dai.node.VideoEncoder)
    still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    rgb_preview.link(still_encoder.input)

    window_name = f"[{friendly_id + 1}] Camera - mxid: {device.getMxId()}"
    manager.set_custom_key(window_name)

    pipeline.create(Display, callback_frame, callback_params, window_name, device, friendly_id).build(
        cam_preview=rgb_preview,
        cam_still=still_encoder.bitstream,
        ctrl_queue=cam_rgb.inputControl.createInputQueue()
    )

    return pipeline


def pair_device_with_pipeline(dev_info : dai.DeviceInfo, pipelines : list, callback_frame : callable, 
                              callback_params : callable, friendly_id : int) -> None:

    device: dai.Device = dai.Device(dev_info)
    print("=== Connected to " + dev_info.getMxId())
    pipelines.append(get_pipelines(device, callback_frame, callback_params, friendly_id))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")


pipelines : list[dai.Pipeline] = []
threads : list[threading.Thread] = []
manager = OpencvManager()

for friendly_id, dev in enumerate(devices):
    pair_device_with_pipeline(dev, pipelines, manager.set_frame, manager.set_params, friendly_id)

for pipeline in pipelines:
    thread = threading.Thread(target=run_pipeline, args=(pipeline,))
    thread.start()
    threads.append(thread)

manager.run()

for pipeline in pipelines:
    pipeline.stop()

for thread in threads:
    thread.join()

print("Devices closed")
