import cv2
import depthai as dai
from birdseyeview import BirdsEyeView
from camera import Camera
import config
import threading
import numpy as np


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
                        return

                    # TOGGLE DEPTH VIEW - press `d` to toggle depth view
                    if key == ord('d'):
                        for camera in cameras:
                            camera.show_detph = not camera.show_detph

                    for camera in cameras:
                        camera.update()

                    birds_eye_view.render()
                    

    def set_frame(self, frame : dai.ImgFrame, window_name : str) -> None:
        with self.lock:
            self.frames[window_name] = frame
            self.newFrameEvent.set()

    
    def set_params(self) -> None:
        pass


def filter_internal_cameras(devices : list[dai.DeviceInfo]) -> list[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def run_pipeline(pipeline : dai.Pipeline) -> None:
    pipeline.run()


def get_pipelines(device : dai.Device, callback_frame : callable, callback_params : callable, friendly_id : int) -> dai.Pipeline:
    pipeline = dai.Pipeline(device)

    # RGB cam -> 'rgb'
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setPreviewSize(640, 360)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewKeepAspectRatio(False)

    # Still encoder -> 'still'
    still_encoder = pipeline.create(dai.node.VideoEncoder)
    still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    cam_rgb.still.link(still_encoder.input)

    window_name = f"[{friendly_id + 1}] Camera - mxid: {device.getMxId()}"
    manager.set_custom_key(window_name)

    pipeline.create(Display, callback_frame, callback_params, window_name, device, friendly_id).build(
        cam_preview=cam_rgb.preview,
        cam_still=still_encoder.bitstream,
        ctrl_queue=cam_rgb.inputControl.createInputQueue()
    )

    return pipeline


def pair_device_with_pipeline(dev_info : dai.DeviceInfo, pipelines : list, callback_frame : callable, 
                              callback_params : callable, friendly_id : int) -> None:

    device: dai.Device = dai.Device(dev_info)

    print("=== Connected to " + dev_info.getMxId())

    pipelines.append(get_pipelines(device, callback_frame, callback_params,friendly_id))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")

devices.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

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

# birds_eye_view = BirdsEyeView(cameras, config.size[0], config.size[1], config.scale)
