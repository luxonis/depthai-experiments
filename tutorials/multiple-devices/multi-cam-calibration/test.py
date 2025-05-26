import os
import depthai as dai
from typing import List, Optional, Dict, Any
import numpy as np
import config
from utils.utility import (
    filter_devices,
    setup_devices,
    start_pipelines,
    any_pipeline_running,
)
from utils.arguments import initialize_argparser
from utils.calibration_node import CalibrationNode


_, args = initialize_argparser()

HTTP_PORT = 8082

def setup_device_pipeline(
    dev_info: dai.DeviceInfo, visualizer: dai.RemoteConnection
) -> Optional[Dict[str, Any]]:
    mxid = dev_info.getDeviceId()
    print(f"\nAttempting to connect to device: {mxid}...")
    device_instance = dai.Device(dev_info)

    print(f"=== Successfully connected to device: {mxid}")

    cameras = device_instance.getConnectedCameraFeatures()
    print(f"    >>> Cameras: {[c.socket.name for c in cameras]}")
    print(f"    >>> USB speed: {device_instance.getUsbSpeed().name}")

    pipeline = dai.Pipeline(device_instance)
    print(f"    Pipeline created for device: {mxid}")

    cam_preview = (
        pipeline.create(dai.node.Camera)
        .build(dai.CameraBoardSocket.CAM_A)
        .requestOutput((640, 400), dai.ImgFrame.Type.NV12)
    )

    intrinsic_mat = np.array(
        device_instance.readCalibration().getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, 3840, 2160
        )
    )

    calibration_node = pipeline.create(CalibrationNode).build(
        cam_preview,
        intrinsic_mat,
        device_instance,
    )

    visualizer.addTopic(f"{mxid}", cam_preview)
    visualizer.addTopic(f"{mxid} calibration annotations", calibration_node.annotation_out)

    print(f"    Pipeline for {mxid} configured. Ready to be started.")
    return {"device": device_instance, "pipeline": pipeline, "mxid": mxid, "calibration_node": calibration_node}


def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    available_devices_info = filter_devices(
        all_device_infos,
        include_ip=args.include_ip,
        max_devices=args.max_devices,
    )

    if not available_devices_info:
        print("No DepthAI devices found.")
        return

    print(f"Found {len(available_devices_info)} DepthAI devices to configure.")

    visualizer = dai.RemoteConnection(httpPort=HTTP_PORT)

    initialized_setups: List[Dict[str, Any]] = setup_devices(
        available_devices_info, visualizer, setup_device_pipeline
    )
    if not initialized_setups:
        print("No devices were successfully set up. Exiting.")
        return

    active_pipelines_info: List[Dict[str, Any]] = start_pipelines(
        initialized_setups, visualizer
    )
    print(f"\n{len(active_pipelines_info)} device(s) should be streaming.")

    current_idx = 0
    while any_pipeline_running(active_pipelines_info):
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got 'q' key from the remote connection! Shutting down.")
            os._exit(0)
        elif key == ord('a'):
            current_idx = (current_idx - 1) % len(active_pipelines_info)
            for setup in active_pipelines_info:
                setup["calibration_node"].calib = False
            active_pipelines_info[current_idx]["calibration_node"].calib = True
        elif key == ord('c'):


if __name__ == "__main__":
    main()
