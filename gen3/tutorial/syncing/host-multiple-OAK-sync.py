#!/usr/bin/env python3

import depthai as dai
import argparse
import threading
from multiple_devices_utilities import OpencvManager
from multiple_devices_utilities import filter_internal_cameras


parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')

args = parser.parse_args()


def run_pipeline(pipeline : dai.Pipeline) -> None:
    pipeline.run()


def get_pipeline(device : dai.Device, stereo : bool, callback : callable) -> dai.Pipeline:
    pipeline = dai.Pipeline(device)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 3)

    if stereo:
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(args.fps)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setFps(args.fps)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        pipeline.create(DisplayStereo, callback, device.getMxId()).build(
            d_mxid=device.getMxId(),
            rgb_out=cam_rgb.preview,
            left_out=mono_left.out,
            right_out=mono_right.out,
        )
    else:
        pipeline.create(Display, callback, device.getMxId()).build(
            d_mxid=device.getMxId(),
            rgb_out=cam_rgb.preview,
        )

    return pipeline


def pair_device_with_pipeline(dev_info, pipelines : list, callback : callable) -> None:
    device: dai.Device = dai.Device(dev_info)

    stereo = len(device.getConnectedCameras())==3

    if stereo: 
        manager.set_custom_keys(["left - "+ str(device.getMxId()), 
                                "rgb - " + str(device.getMxId()),
                                "right - " + str(device.getMxId())])   
    else: 
        manager.set_custom_keys(["rgb - " + str(device.getMxId())])

    pipelines.append(get_pipeline(device, stereo, callback))


class Display(dai.node.HostNode):
    def __init__(self, frame_callback : callable, d_mxid : str) -> None:
        super().__init__()
        self.callback = frame_callback
        self.d_mxid = d_mxid


    def build(self, rgb_out : dai.Node.Output, d_mxid : str) -> "Display":
        
        self.d_mxid = d_mxid
        self.link_args(rgb_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgb_frame : dai.ImgFrame) -> None:

        self.callback(rgb_frame.getCvFrame(), "rgb - " + str(self.d_mxid))


class DisplayStereo(dai.node.HostNode):
    def __init__(self, frame_callback : callable, d_mxid : str) -> None:
        super().__init__()
        self.callback = frame_callback
        self.d_mxid = d_mxid


    def build(self, rgb_out : dai.Node.Output, d_mxid : str, left_out : dai.Node.Output, 
              right_out : dai.Node.Output) -> "DisplayStereo":
        
        self.d_mxid = d_mxid

        self.link_args(rgb_out, left_out, right_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgb_frame : dai.ImgFrame, left_frame : dai.ImgFrame, 
                right_frame : dai.ImgFrame) -> None:

        self.callback(left_frame.getCvFrame(), "right - " + str(self.d_mxid))
        self.callback(right_frame.getCvFrame(), "left - " + str(self.d_mxid))
        self.callback(rgb_frame.getCvFrame(), "rgb - " + str(self.d_mxid))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())

if len(devices) == 0: raise RuntimeError("No devices found!")
else: print("Found", len(devices), "devices")


pipelines : list[dai.Pipeline] = []
threads : list[threading.Thread] = []
manager = OpencvManager()

for dev in devices:
    pair_device_with_pipeline(dev, pipelines, manager.set_frame)

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
