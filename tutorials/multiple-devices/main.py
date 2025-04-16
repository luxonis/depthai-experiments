import depthai as dai
import threading
import cv2
from utility import filter_internal_cameras, run_pipeline
from typing import Callable, List


class OpencvManager:
    def __init__(self, keys: List[int]):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.frames = self._init_frames(keys)

    def run(self) -> None:
        while True:
            self.newFrameEvent.wait()
            for dx_id in self.frames.keys():
                if self.frames[dx_id] is not None:
                    cv2.imshow(f"rgb - {dx_id}", self.frames[dx_id])

                    if cv2.waitKey(1) == ord("q"):
                        return

    def setFrame(self, frame: dai.ImgFrame, dx_id: int) -> None:
        with self.lock:
            self.frames[dx_id] = frame
            self.newFrameEvent.set()

    def _init_frames(self, keys: List[int]) -> dict:
        dic = dict()
        for key in keys:
            dic[key] = None
        return dic


class Display(dai.node.HostNode):
    def __init__(self, frameCallback: Callable, dx_id: int) -> None:
        super().__init__()
        self.callback = frameCallback
        self.dx_id = dx_id

    def build(self, cam_out: dai.Node.Output) -> "Display":
        self.link_args(cam_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, in_frame: dai.ImgFrame) -> None:
        self.callback(in_frame.getCvFrame(), self.dx_id)


def getPipeline(dev: dai.Device, callback: Callable) -> dai.Pipeline:
    pipeline = dai.Pipeline(dev)
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    rgb_preview = cam_rgb.requestOutput(size=(600, 300))

    pipeline.create(Display, callback, dev.getMxId()).build(cam_out=rgb_preview)

    return pipeline


def pair_device_with_pipeline(
    dev_info: dai.DeviceInfo, pipelines: List, callback: Callable
) -> None:
    device: dai.Device = dai.Device(dev_info)

    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    pipelines.append(getPipeline(device, callback))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
print(f"Found {len(devices)} internal devices")


pipelines: List[dai.Pipeline] = []
threads: List[threading.Thread] = []
manager = OpencvManager([device.getMxId() for device in devices])

for dev in devices:
    pair_device_with_pipeline(dev, pipelines, manager.setFrame)

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
