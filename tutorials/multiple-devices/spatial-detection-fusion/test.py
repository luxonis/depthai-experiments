import os
import depthai as dai
from typing import List, Optional, Dict, Any, Callable, Tuple
import numpy as np

from utils.utility import (
    filter_devices,
    setup_devices,
    start_pipelines,
    any_pipeline_running # This will primarily check device pipelines
)
from utils.arguments import initialize_argparser
from utils.birds_eye_view import BirdsEyeView 
from utils import config as app_config

_, args = initialize_argparser()

def load_extrinsics_for_all_devices(
    devices_info: List[dai.DeviceInfo]
) -> Dict[str, Dict[str, Any]]:
    """Loads extrinsic calibration data (cam_to_world matrix) and assigns friendly IDs."""
    all_extrinsics: Dict[str, Dict[str, Any]] = {}
    calibration_dir = app_config.calibration_data_dir
    
    for friendly_id_counter, dev_info in enumerate(devices_info):
        mxid = dev_info.getDeviceId()
        try:
            file_path = os.path.join(os.getcwd(), calibration_dir, f"extrinsics_{mxid}.npz")
            if os.path.exists(file_path):
                data = np.load(file_path)
                if 'cam_to_world' in data:
                    all_extrinsics[mxid] = {
                        'cam_to_world': data['cam_to_world'],
                        'friendly_id': friendly_id_counter + 1 
                    }
                    print(f"Successfully loaded extrinsics for {mxid} (Friendly ID: {friendly_id_counter + 1})")
            else:
                print(f"Warning: Extrinsics file not found for {mxid} at {file_path}.")
        except Exception as e:
            print(f"Error loading extrinsics for {mxid}: {e}.")
    return all_extrinsics

def setup_device_pipeline(
    dev_info: dai.DeviceInfo,
    visualizer: dai.RemoteConnection, 
) -> Optional[Dict[str, Any]]:
    """Sets up the DepthAI pipeline for a single device for BEV application."""
    mxid = dev_info.getDeviceId()
    try:
        device_instance = dai.Device(dev_info)
        print(f"    === Successfully connected to device: {mxid}")
    except RuntimeError as e:
        print(f"    ERROR: Failed to connect to device {mxid}: {e}")
        return None

    pipeline = dai.Pipeline(device_instance)
    print(f"        Pipeline created for device: {mxid}")

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    left_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=args.fps_limit
    )
    right_cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=args.fps_limit
    )
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput((640, 400)),
        right=right_cam.requestOutput((640, 400)),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    platform = dev_info.platform
    if platform == "RVC2":
        stereo.setOutputSize(*nn_archive.getInputSize())
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)

    model_description = dai.NNModelDescription(app_config.nn_model_slug, platform=platform)
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description, useCached=False))

    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=cam, stereo=stereo, nnArchive=nn_archive, fps=args.fps_limit
    )
    nn.setBoundingBoxScaleFactor(0.5)
    if platform == "RVC2":
        nn.setNNArchive(
            nn_archive, numShaves=6
        )
    labels = nn_archive.getConfig().model.heads[0].metadata.classes
    label_indices = list(range(len(labels))) # Use all labels by default

    object_tracker = pipeline.create(dai.node.ObjectTracker)
    object_tracker.setDetectionLabelsToTrack(label_indices)
    object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    
    visualizer.addTopic(f"{mxid}", nn.passthrough, group=mxid)
    visualizer.addTopic(f"{mxid} - Detections", nn.out, group=mxid)

    print(f"        Device pipeline for {mxid} configured.")
    return {
        "device": device_instance,
        "pipeline": pipeline,
        "mxid": mxid,
        "tracker_out": object_tracker.out,
    }

def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    available_devices_info = filter_devices(
        all_device_infos, include_ip=args.include_ip, max_devices=args.max_devices
    )

    if not available_devices_info:
        print("No DepthAI devices found.")
        return

    print(f"Found {len(available_devices_info)} DepthAI devices to configure.")

    all_cam_extrinsics = load_extrinsics_for_all_devices(available_devices_info)
    if not all_cam_extrinsics: print("No extrinsic calibrations loaded. BEV cannot function. Exiting."); return
    
    devices_to_setup_info = [dev for dev in available_devices_info if dev.getDeviceId() in all_cam_extrinsics]
    print(f"Proceeding with {len(devices_to_setup_info)} devices that have extrinsics.")

    visualizer = dai.RemoteConnection(httpPort=app_config.HTTP_PORT)


    initialized_setups: List[Dict[str, Any]] = setup_devices(
        devices_to_setup_info, visualizer, setup_device_pipeline
    )
    if not initialized_setups: print("No devices were successfully set up. Exiting."); return

    all_trackers_outputs = [setup["tracker_out"] for setup in initialized_setups]

    # Create a single Host pipeline for the BEV node
    host_bev_pipeline = initialized_setups[0]["pipeline"] # Use the first device's pipeline as a base
    bev_node_instance = host_bev_pipeline.create(BirdsEyeView).build(
        all_cam_extrinsics, 
        app_config,
        all_trackers_outputs
    )

    active_pipelines_info: List[Dict[str, Any]] = start_pipelines(initialized_setups, visualizer)
    if not active_pipelines_info: print("No device pipelines are active. Exiting."); return
    print(f"\n{len(active_pipelines_info)} device pipeline(s) started.")

    visualizer.addTopic("Bird's Eye View", output=bev_node_instance.output, group="World View")
    print("BirdsEyeView created and its output topic added to visualizer.")

    while any_pipeline_running(active_pipelines_info):
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got 'q' key from the remote connection! Shutting down.")
            os._exit(0)

if __name__ == "__main__":
    main()

