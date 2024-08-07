import depthai as dai
import threading
import config
from typing import List
from host_pointcloud import PointCloud
from threaded_display import PointCloudVisualizer
from threaded_display import OpencvManager
from utility import filter_internal_cameras, run_pipeline


def get_pipelines(device : dai.Device, callback_frame : callable, friendly_id : int) -> dai.Pipeline:
    pipeline = dai.Pipeline(device)

    mono_left = pipeline.create(dai.node.MonoCamera) 
    mono_right = pipeline.create(dai.node.MonoCamera) 
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    
    cam_stereo = pipeline.create(dai.node.StereoDepth).build(left=mono_left.out, right=mono_right.out)
    cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    cam_stereo.initialConfig.setMedianFilter(config.median)
    cam_stereo.initialConfig.setConfidenceThreshold(config.confidence_threshold)
    cam_stereo.setLeftRightCheck(config.lrcheck)
    cam_stereo.setExtendedDisparity(config.extended)
    cam_stereo.setSubpixel(config.subpixel)
    cam_stereo.setSubpixelFractionalBits(3) 

    init_config = cam_stereo.initialConfig
    init_config.postProcessing.speckleFilter.enable = False
    init_config.postProcessing.speckleFilter.speckleRange = 50
    init_config.postProcessing.temporalFilter.enable = True
    init_config.postProcessing.spatialFilter.enable = True
    init_config.postProcessing.spatialFilter.holeFillingRadius = 2
    init_config.postProcessing.spatialFilter.numIterations = 1
    init_config.postProcessing.thresholdFilter.minRange = config.min_range
    init_config.postProcessing.thresholdFilter.maxRange = config.max_range
    init_config.postProcessing.decimationFilter.decimationFactor = 1

    if config.COLOR:
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setIspScale(1, 3)
        cam_rgb.initialControl.setManualFocus(130)
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        image_size = cam_rgb.getIspSize()
    else:
        image_size = mono_right.getResolutionSize()
    

    window_name = f"[{friendly_id + 1}] Camera - mxid: {device.getMxId()}"
    manager.set_custom_key(window_name, device.getMxId())
    manager.set_params(PCLvisualizer.point_cloud_window)

    pipeline.create(PointCloud, callback_frame, window_name, device, image_size, PCLvisualizer).build(
        depth_out=cam_stereo.depth,
        cam_isp=cam_rgb.isp if config.COLOR else cam_stereo.rectifiedRight
    )

    return pipeline


def pair_device_with_pipeline(dev_info : dai.DeviceInfo, pipelines : List, callback_frame : callable, 
                              friendly_id : int) -> None:

    device: dai.Device = dai.Device(dev_info)
    print("=== Connected to " + dev_info.getMxId())
    pipelines.append(get_pipelines(device, callback_frame, friendly_id))


devices = filter_internal_cameras(dai.Device.getAllAvailableDevices())
if len(devices) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(devices), "devices")


pipelines : List[dai.Pipeline] = []
threads : List[threading.Thread] = []
manager = OpencvManager()
PCLvisualizer = PointCloudVisualizer()

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