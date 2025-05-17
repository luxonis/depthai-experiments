import depthai as dai
from typing import List, Optional
from util.utility import filter_internal_cameras


HTTP_PORT = 8082

def setup_device_pipeline(
    dev_info: dai.DeviceInfo,
    visualizer: dai.RemoteConnection
) -> Optional[dai.Device]:
    device_instance = None
    try:
        device_instance = dai.Device(dev_info)
        mxid = device_instance.getDeviceId() 

        print(f"=== Configuring device: {mxid}")
        cameras = device_instance.getConnectedCameraFeatures()
        print(f"    >>> Cameras: {[c.socket.name for c in cameras]}")
        print(f"    >>> USB speed: {device_instance.getUsbSpeed().name}")

        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        rgb_preview = cam_rgb.requestOutput(size=(600, 300))

        visualizer.addTopic(f"RGB - {mxid}", rgb_preview)

        visualizer.registerPipeline(pipeline)
        device_instance.startPipeline(pipeline)

        print(f"    >>> Pipeline configured, registered, and started for {mxid}")
        return device_instance

    except RuntimeError as e:
        print(f"Error setting up device {dev_info.getDeviceId()}: {e}")
        if device_instance:
            device_instance.close() # Ensure device is closed on error
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred with device {dev_info.getDeviceId()}: {e}")
        if device_instance:
            device_instance.close()
        return None


def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    available_devices_info = filter_internal_cameras(all_device_infos)

    if not available_devices_info:
        print("No DepthAI devices found or filtered out.")
        return

    print(f"Found {len(available_devices_info)} DepthAI devices to configure.")

    visualizer = dai.RemoteConnection(httpPort=HTTP_PORT)

    active_devices: List[dai.Device] = []

    for dev_info in available_devices_info:
        device = setup_device_pipeline(dev_info, visualizer)
        if device:
            active_devices.append(device)

    if not active_devices:
        print("No devices were successfully initialized and started. Exiting.")
        return

    print("\nAll configured devices should be streaming.")
    print(f"Open your browser to http://localhost:{HTTP_PORT} (or the configured port)")
    print("Press 'q' in the visualizer window (if it captures keys) or Ctrl+C in this console to quit.")

    try:
        while True:
            key = visualizer.waitKey(1)
            if key == ord('q'):
                print("Got q key from the remote connection!")
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        print("Closing down application...")
        for device in active_devices:
            try:
                print(f"Closing device {device.getDeviceId()}...")
                device.close()
            except Exception as e:
                print(f"Error closing a device: {e}")
        print("Application terminated.")

if __name__ == "__main__":
    main()
