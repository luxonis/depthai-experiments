import os
import time
import depthai as dai
from typing import List, Optional, Dict, Any
import numpy as np
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
STILL_WIDTH, STILL_HEIGHT = 3840, 2160 # 4K UHD

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

    preview_width, preview_height = (640, 400)
    cam_socket = dai.CameraBoardSocket.CAM_A
    camera_object = pipeline.create(dai.node.Camera).build(cam_socket)
    cam_preview = camera_object.requestOutput(
        (preview_width, preview_height), dai.ImgFrame.Type.NV12
    )

    calib_data = device_instance.readCalibration()
    intrinsic_mat_still = np.array(
        calib_data.getCameraIntrinsics(cam_socket, STILL_WIDTH, STILL_HEIGHT)
    )
    intrinsic_mat_preview = np.array(
        calib_data.getCameraIntrinsics(cam_socket, preview_width, preview_height)
    )

    calibration_node = pipeline.create(CalibrationNode).build(
        preview_input=cam_preview, 
        intrinsic_mat_still=intrinsic_mat_still,
        intrinsic_mat_preview=intrinsic_mat_preview,
        device=device_instance,
        preview_width=preview_width,
        preview_height=preview_height
        still_width=STILL_WIDTH,
        still_height=STILL_HEIGHT,
    )

    high_res_still_output = camera_object.requestOutput(
        (STILL_WIDTH, STILL_HEIGHT), dai.ImgFrame.Type.NV12
    )

    visualizer.addTopic(f"{mxid}", cam_preview)
    visualizer.addTopic(f"{mxid} calibration annotations", calibration_node.annotation_out)

    control_q = camera_object.inputControl.createInputQueue()
    # still_q = still_encoder.bitstream.createOutputQueue(maxSize=2, blocking=False)
    still_q = high_res_still_output.createOutputQueue(maxSize=2, blocking=False)

    print(f"    Pipeline for {mxid} configured. Ready to be started.")
    return {
        "device": device_instance,
        "pipeline": pipeline,
        "mxid": mxid,
        "calibration_node": calibration_node,
        "control_queue": control_q, 
        "still_queue": still_q,
    }


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
    if active_pipelines_info:
        # Initially select the first camera for calibration
        active_pipelines_info[current_idx]["calibration_node"].calib = True


    while any_pipeline_running(active_pipelines_info): 
        key = visualizer.waitKey(1)

        if not active_pipelines_info: # If all pipelines stopped or failed
            print("No active pipelines. Exiting loop.")
            break

        if key == ord("q"):
            print("Got 'q' key from the remote connection! Shutting down.")
            for setup in active_pipelines_info:
                if setup.get("device"):
                    setup["device"].close()
            os._exit(0) 

        elif key == ord('a'): 
            active_pipelines_info[current_idx]["calibration_node"].calib = False
            current_idx = (current_idx + 1) % len(active_pipelines_info) # Cycle through cameras
            active_pipelines_info[current_idx]["calibration_node"].calib = True
            print(f"INFO: Switched calibration focus to device {active_pipelines_info[current_idx]['mxid']}")

        elif key == ord('c'): 
            if not active_pipelines_info: continue

            current_setup = active_pipelines_info[current_idx]

            control_q = current_setup.get("control_queue")
            still_q = current_setup.get("still_queue")

            # 1. Send capture still command
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            print(f"[{current_setup['mxid']}] Sending capture still command...")
            control_q.send(ctrl)

            # 2. Wait for and retrieve the still image
            print(f"[{current_setup['mxid']}] Waiting for still image...")
            start_time_capture = time.monotonic()
            timeout_sec_capture = 3.0  # Timeout for receiving the still image
            in_still_msg = None

            # Empty the queue of any old stills first
            while still_q.tryGet() is not None: pass 

            while time.monotonic() - start_time_capture < timeout_sec_capture:
                in_still_msg = still_q.tryGet() # Non-blocking get
                if in_still_msg is not None:
                    print(f"[{current_setup['mxid']}] Still image received.")
                    break
                time.sleep(0.05) # Brief pause to avoid busy-waiting too aggressively

            if in_still_msg:
                try:
                    raw_cv_frame = in_still_msg.getCvFrame()
                    
                    frame_type = in_still_msg.getType()
                    frame_height_meta = in_still_msg.getHeight()
                    frame_width_meta = in_still_msg.getWidth()
                    
                    data_shape = raw_cv_frame.shape
                    
                    print(f"[{current_setup['mxid']}] ImgFrame received (reported type: {frame_type}, " +
                          f"reported size: {frame_width_meta}x{frame_height_meta}, data_shape: {data_shape}).")

                    still_frame_bgr = raw_cv_frame 
                    current_setup['calibration_node'].trigger_calibration_with_image(still_frame_bgr)
                except Exception as e:
                    print(f"[{current_setup['mxid']}] ERROR processing raw still image: {e}")
            else:
                print(f"[{current_setup['mxid']}] WARNING: Timeout - Did not receive still image within {timeout_sec_capture}s.")


    print("INFO: Main loop finished. Cleaning up...")
    for setup in active_pipelines_info:
        if setup.get("device"):
            try:
                setup["device"].close()
            except Exception as e:
                print(f"Error closing device {setup.get('mxid', 'N/A')}: {e}")


if __name__ == "__main__":
    main()
