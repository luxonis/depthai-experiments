import depthai as dai
from typing import List, Optional, Dict, Any
from utils.utility import filter_internal_cameras

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

    visualizer.addTopic(f"RGB - {mxid}", cam_preview)

    print(f"    Pipeline for {mxid} configured. Ready to be started.")
    return {"device": device_instance, "pipeline": pipeline, "mxid": mxid}


def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    available_devices_info = filter_internal_cameras(all_device_infos)

    if not available_devices_info:
        print("No DepthAI devices found.")
        return

    print(f"Found {len(available_devices_info)} DepthAI devices to configure.")

    visualizer = dai.RemoteConnection(httpPort=HTTP_PORT)

    initialized_setups: List[Dict[str, Any]] = []

    for dev_info in available_devices_info:
        setup_info = setup_device_pipeline(dev_info, visualizer)
        if setup_info:
            initialized_setups.append(setup_info)
        else:
            mxid_for_error = (
                dev_info.getMxId()
                if hasattr(dev_info, "getMxId")
                else dev_info.getDeviceId()
            )
            print(f"--- Failed to set up device {mxid_for_error}. Skipping. ---")

    if not initialized_setups:
        print("No devices were successfully set up. Exiting.")
        return

    active_pipelines_info: List[Dict[str, Any]] = []
    for setup in initialized_setups:
        try:
            print(f"\nStarting pipeline for device {setup['mxid']}...")
            setup["pipeline"].start()

            visualizer.registerPipeline(setup["pipeline"])
            print(f"Pipeline for {setup['mxid']} registered with visualizer.")
            active_pipelines_info.append(setup)
        except Exception as e:
            print(
                f"Error starting or registering pipeline for device {setup['mxid']}: {e}"
            )
            setup["device"].close()

    print(f"\n{len(active_pipelines_info)} device(s) should be streaming.")

    while True:
        all_running = True
        for item_info in active_pipelines_info:
            if not item_info["pipeline"].isRunning():
                print(f"Pipeline for device {item_info['mxid']} is no longer running!")
                all_running = False
        if not all_running and not active_pipelines_info:
            print("All active pipelines have stopped.")
            break

        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got 'q' key from the remote connection! Shutting down.")
            break
        if not active_pipelines_info:
            break


if __name__ == "__main__":
    main()
