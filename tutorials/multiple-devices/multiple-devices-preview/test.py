import depthai as dai
from typing import List, Optional, Dict, Any # Added Dict, Any
import time # For potential delays if needed

HTTP_PORT = 8082

def setup_device_pipeline(
    dev_info: dai.DeviceInfo,
    visualizer: dai.RemoteConnection
) -> Optional[Dict[str, Any]]: # Return a dictionary with device, pipeline, mxid
    device_instance = None
    mxid = dev_info.getMxId() if hasattr(dev_info, 'getMxId') else dev_info.getDeviceId()

    try:
        print(f"Attempting to connect to device: {mxid}...")
        device_instance = dai.Device(dev_info)
        mxid = device_instance.getMxId() if hasattr(device_instance, 'getMxId') else device_instance.getDeviceId()
        print(f"=== Successfully connected to device: {mxid}")

        cameras = device_instance.getConnectedCameraFeatures()
        print(f"    >>> Cameras: {[c.socket.name for c in cameras]}")
        print(f"    >>> USB speed: {device_instance.getUsbSpeed().name}")

        pipeline = dai.Pipeline(device_instance)
        print(f"    Pipeline created for device: {mxid}")

        cam_preview = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A).requestOutput((640,400), dai.ImgFrame.Type.NV12)
        
        visualizer.addTopic(f"RGB - {mxid}", cam_preview)

        print(f"    Pipeline for {mxid} configured. Ready to be started.")
        return {'device': device_instance, 'pipeline': pipeline, 'mxid': mxid}

    except RuntimeError as e:
        print(f"Error during setup for device {mxid}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with device {mxid}: {e}")
        return None

def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    # available_devices_info = filter_internal_cameras(all_device_infos) # Your filter
    available_devices_info = all_device_infos

    if not available_devices_info:
        print("No DepthAI devices found.")
        return

    print(f"Found {len(available_devices_info)} DepthAI devices to configure.")

    visualizer = dai.RemoteConnection(httpPort=HTTP_PORT)
    print(f"To connect to the DepthAI visualizer, open http://localhost:{HTTP_PORT} in your browser")

    initialized_setups: List[Dict[str, Any]] = []

    for dev_info in available_devices_info:
        # Optional: Add a small delay if rapidly initializing devices causes issues
        time.sleep(0.5) 
        
        setup_info = setup_device_pipeline(dev_info, visualizer)
        if setup_info:
            initialized_setups.append(setup_info)
        else:
            mxid_for_error = dev_info.getMxId() if hasattr(dev_info, 'getMxId') else dev_info.getDeviceId()
            print(f"--- Failed to set up device {mxid_for_error}. Skipping. ---")

    if not initialized_setups:
        print("No devices were successfully set up. Exiting.")
        return

    active_pipelines_info: List[Dict[str, Any]] = []
    for setup in initialized_setups:
        try:
            print(f"\nStarting pipeline for device {setup['mxid']}...")
            setup['pipeline'].start() 
            print(f"Pipeline for {setup['mxid']} started.")
            
            print(f"Registering pipeline for {setup['mxid']} with visualizer...")
            visualizer.registerPipeline(setup['pipeline'])
            print(f"Pipeline for {setup['mxid']} registered with visualizer.")
            active_pipelines_info.append(setup) 
        except Exception as e:
            print(f"Error starting or registering pipeline for device {setup['mxid']}: {e}")
            try:
                setup['device'].close()
            except Exception as e_close:
                print(f"Further error closing device {setup['mxid']} after start/register failure: {e_close}")
    
    if not active_pipelines_info:
        print("No pipelines were successfully started and registered. Exiting.")
        return

    print(f"\n{len(active_pipelines_info)} device(s) should be streaming.")
    print("Press 'q' in the visualizer window (if it captures keys) or Ctrl+C in this console to quit.")

    try:
        while True:
            all_running = True
            for item_info in active_pipelines_info:
                if not item_info['pipeline'].isRunning(): # Assuming isRunning() exists
                    print(f"Pipeline for device {item_info['mxid']} is no longer running!")
                    all_running = False # Or handle this more gracefully, e.g., attempt restart or remove
            if not all_running and not active_pipelines_info: # if all stopped or list became empty
                print("All active pipelines have stopped.")
                break

            key = visualizer.waitKey(1)
            if key == ord('q'):
                print("Got 'q' key from the remote connection! Shutting down.")
                break
            if not active_pipelines_info:
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        print("Closing down application...")
        for item_info in initialized_setups: # Try to close all initially setup devices
            try:
                print(f"Closing device {item_info['mxid']}...")
                item_info['device'].close()
            except Exception as e:
                print(f"Error closing device {item_info.get('mxid', 'Unknown')}: {e}")
        print("Application terminated.")

if __name__ == "__main__":
    main()
