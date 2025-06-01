import cv2
import depthai as dai
import threading
import numpy as np
from detection import Detection
import os
import config
from birdseyeview import BirdsEyeView
from typing import List, Dict, Callable

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

label_map = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class OpencvManager:
    def __init__(self):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.keys = []
        self.frames: Dict[str, dai.ImgFrame] = {}  # window_name -> frame
        self.detections: Dict[
            str, dai.SpatialImgDetections
        ] = {}  # window_name -> detections
        self.depth_frames: Dict[str, np.ndarray] = {}  # window_name -> depth_frame
        self.dx_ids: Dict[str, str] = {}  # window_name -> device_id
        self.show_detph = False
        self.detected_objects: List[Detection] = []
        self.cam_to_world: Dict[str, any] = {}  # device_id -> cam_to_world
        self.friendly_id: Dict[str, int] = {}  # device_id -> friendly_id

    def run(self) -> None:
        for window_name in self.frames.keys():
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 360)

        self._load_calibration()
        self._birds_eye_view = BirdsEyeView(
            self.cam_to_world,
            self.friendly_id,
            config.size[0],
            config.size[1],
            config.scale,
        )

        while True:
            self.newFrameEvent.wait()
            for name in self.frames.keys():
                if (
                    self.frames[name] is not None
                    and self.depth_frames[name] is not None
                ):
                    key = cv2.waitKey(1)

                    # QUIT - press `q` to quit
                    if key == ord("q"):
                        for dx_id in self.dx_ids.values():
                            print("=== Closed " + dx_id)
                        return

                    # TOGGLE DEPTH VIEW - press `d` to toggle depth view
                    if key == ord("d"):
                        self.show_detph = not self.show_detph

                    self._update(
                        self.frames[name],
                        self.detections[name],
                        self.depth_frames[name],
                        name,
                    )

                    self._birds_eye_view.render(self.detected_objects)

    def set_frame(
        self,
        frame: dai.ImgFrame,
        detections: dai.SpatialImgDetections,
        depth_frame: np.ndarray,
        window_name: str,
    ) -> None:
        with self.lock:
            self.frames[window_name] = frame
            self.detections[window_name] = detections
            self.depth_frames[window_name] = depth_frame
            self.newFrameEvent.set()

    def set_params(self, window_name: str, dx_id: str, friendly_id: int) -> None:
        self.dx_ids[window_name] = dx_id
        self.friendly_id[dx_id] = friendly_id + 1

    def set_custom_key(self, key: str) -> None:
        self.keys.append(key)
        self._init_frames()

    def _init_frames(self) -> None:
        for key in self.keys:
            if key not in self.frames.keys():
                self.frames[key] = None

    def _load_calibration(self):
        path = os.path.join(os.path.dirname(__file__), f"{config.calibration_data_dir}")
        try:
            for dx_id in self.dx_ids.values():
                extrinsics = np.load(f"{path}/extrinsics_{dx_id}.npz")
                self.cam_to_world[dx_id] = extrinsics["cam_to_world"]
        except Exception as _:
            raise RuntimeError(
                f"Could not load calibration data for camera {dx_id} from {path}!"
            )

    def _update(
        self,
        in_rgb: dai.ImgFrame,
        in_nn: dai.SpatialImgDetections,
        in_depth: dai.ImgFrame,
        window_name: str,
    ) -> None:
        depth_frame = in_depth.getFrame()  # depthFrame values are in millimeters
        depth_frame_color = cv2.normalize(
            depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
        )
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

        self.frame_rgb = in_rgb.getCvFrame()

        if self.show_detph:
            visualization = depth_frame_color.copy()
        else:
            visualization = self.frame_rgb.copy()
        visualization = cv2.resize(
            visualization, (640, 360), interpolation=cv2.INTER_NEAREST
        )

        height = visualization.shape[0]
        width = visualization.shape[1]

        detections = []
        if in_nn is not None:
            detections = in_nn.detections

        self.detected_objects = []

        for detection in detections:
            roi = detection.boundingBoxMapping.roi
            roi = roi.denormalize(width, height)
            top_left = roi.topLeft()
            bottom_right = roi.bottomRight()
            xmin = int(top_left.x)
            ymin = int(top_left.y)
            xmax = int(bottom_right.x)
            ymax = int(bottom_right.y)

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = label_map[detection.label]
            except Exception as _:
                label = detection.label

            if self.cam_to_world[self.dx_ids[window_name]] is not None:
                pos_camera_frame = np.array(
                    [
                        [
                            detection.spatialCoordinates.x / 1000,
                            -detection.spatialCoordinates.y / 1000,
                            detection.spatialCoordinates.z / 1000,
                            1,
                        ]
                    ]
                ).T
                # pos_camera_frame = np.array([[0, 0, detection.spatialCoordinates.z/1000, 1]]).T
                pos_world_frame = (
                    self.cam_to_world[self.dx_ids[window_name]] @ pos_camera_frame
                )

                self.detected_objects.append(
                    Detection(
                        label,
                        detection.confidence,
                        pos_world_frame,
                        self.friendly_id[self.dx_ids[window_name]],
                    )
                )

            cv2.rectangle(visualization, (xmin, ymin), (xmax, ymax), (100, 0, 0), 2)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                visualization,
                str(label),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                visualization,
                "{:.2f}".format(detection.confidence * 100),
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                visualization,
                f"X: {int(detection.spatialCoordinates.x)} mm",
                (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                visualization,
                f"Y: {int(detection.spatialCoordinates.y)} mm",
                (x1 + 10, y1 + 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                visualization,
                f"Z: {int(detection.spatialCoordinates.z)} mm",
                (x1 + 10, y1 + 80),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )

        cv2.imshow(window_name, visualization)


class Display(dai.node.HostNode):
    def __init__(self, callback_frame: Callable, window_name: str) -> None:
        super().__init__()
        self.callback_frame = callback_frame
        self.widnow_name = window_name

    def build(
        self,
        cam_rgb: dai.Node.Output,
        nn_out: dai.Node.Output,
        depth_out: dai.Node.Output,
    ) -> "Display":
        self.inputs["rgb_frame"].setBlocking(False)
        self.inputs["rgb_frame"].setMaxSize(1)
        self.inputs["nn_out"].setBlocking(False)
        self.inputs["nn_out"].setMaxSize(1)
        self.inputs["depth_frame"].setBlocking(False)
        self.inputs["depth_frame"].setMaxSize(1)

        self.link_args(cam_rgb, nn_out, depth_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(
        self,
        rgb_frame: dai.ImgFrame,
        nn_out: dai.SpatialImgDetections,
        depth_frame: dai.ImgFrame,
    ) -> None:
        self.callback_frame(rgb_frame, nn_out, depth_frame, self.widnow_name)


def filter_internal_cameras(devices: List[dai.DeviceInfo]) -> List[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def run_pipeline(pipeline: dai.Pipeline) -> None:
    pipeline.run()


def get_pipelines(
    device: dai.Device, callback_frame: Callable, friendly_id: int
) -> dai.Pipeline:
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    preview = cam_rgb.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)

    # Depth cam -> 'depth'
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    cam_stereo = pipeline.create(dai.node.StereoDepth).build(
        left=mono_left.requestOutput(size=(640, 640)),
        right=mono_right.requestOutput(size=(640, 640)),
    )

    cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    cam_stereo.setDepthAlign(
        dai.CameraBoardSocket.CAM_A
    )  # Align depth map to the perspective of RGB camera, on which inference is done
    cam_stereo.setOutputSize(640, 640)

    # Spatial detection network -> 'nn'
    spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    spatial_nn.setBlob(nn_archive.getSuperBlob().getBlobWithNumShaves(7))
    spatial_nn.setConfidenceThreshold(0.6)
    spatial_nn.input.setBlocking(False)
    spatial_nn.setBoundingBoxScaleFactor(0.2)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(5000)

    preview.link(spatial_nn.input)
    cam_stereo.depth.link(spatial_nn.inputDepth)

    window_name = f"[{friendly_id + 1}] Camera - mxid: {device.getMxId()}"
    manager.set_custom_key(window_name)
    manager.set_params(window_name, device.getMxId(), friendly_id)

    pipeline.create(Display, callback_frame, window_name).build(
        cam_rgb=preview, nn_out=spatial_nn.out, depth_out=spatial_nn.passthroughDepth
    )

    return pipeline


def pair_device_with_pipeline(
    dev_info: dai.DeviceInfo,
    pipelines: List,
    callback_frame: Callable,
    friendly_id: int,
) -> None:
    device: dai.Device = dai.Device(dev_info)
    print("=== Connected to " + device.getMxId())
    pipelines.append(get_pipelines(device, callback_frame, friendly_id))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")


pipelines: List[dai.Pipeline] = []
threads: List[threading.Thread] = []
manager = OpencvManager()

for friendly_id, dev in enumerate(devices):
    pair_device_with_pipeline(dev, pipelines, manager.set_frame, friendly_id)

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
