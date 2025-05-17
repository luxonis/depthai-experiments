import depthai as dai
import threading
import cv2
from utils.utility import filter_internal_cameras, run_pipeline
from typing import List, Dict, Callable


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
                    frame = cv2.imdecode(self.frames[dx_id].getData(), cv2.IMREAD_COLOR)
                    frame = cv2.pyrDown(frame)
                    cv2.imshow(dx_id, frame)

                    if cv2.waitKey(1) == ord("q"):
                        return

    def setFrame(self, frame: dai.ImgFrame, dx_id: int) -> None:
        with self.lock:
            self.frames[dx_id] = frame
            self.newFrameEvent.set()

    def _init_frames(self, keys: List[int]) -> Dict:
        dic = dict()
        for key in keys:
            dic[key] = None
        return dic


class DisplayDecodedVideo(dai.node.HostNode):
    def __init__(self, callback: Callable, dx_id: int) -> None:
        super().__init__()
        self.callback = callback
        self.dx_id = dx_id

    def build(self, bitstream_out: dai.Node.Output) -> "DisplayDecodedVideo":
        self.link_args(bitstream_out)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, bitstream: dai.ImgFrame) -> None:
        self.callback(bitstream, self.dx_id)


def getPipeline(dev: dai.Device, callback: Callable) -> dai.Pipeline:
    pipeline = dai.Pipeline(dev)

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_video = camRgb.requestOutput(
        size=(1920, 1080), fps=30, type=dai.ImgFrame.Type.NV12
    )

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    rgb_video.link(videoEnc.input)

    pipeline.create(DisplayDecodedVideo, callback, dev.getMxId()).build(
        bitstream_out=videoEnc.bitstream
    )

    return pipeline


def pair_device_with_pipeline(
    dev_info: dai.DeviceInfo, pipelines: List[dai.Pipeline], callback: Callable
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
print(f"Found {len(devices)} devices")


pipelines: List[dai.Pipeline] = []
manager = OpencvManager([device.getMxId() for device in devices])
threads: List[threading.Thread] = []

for dev in devices:
    pair_device_with_pipeline(dev, pipelines, manager.setFrame)

for pipeline in pipelines:
    thread = threading.Thread(target=run_pipeline, args=(pipeline,))
    thread.start()
    threads.append(thread)

manager.run()

for pipeline in pipelines:
    pipeline.stop()

for t in threads:
    t.join()

print("Devices closed")
