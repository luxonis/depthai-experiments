import os
import depthai as dai
from typing import List, Optional, Dict, Any

from utils.annotation_node import AnnotationNode
from utils.utility import filter_internal_cameras, generate_vibrant_random_color
from depthai_nodes.node import ParsingNeuralNetwork

HTTP_PORT = 8082

LABEL_MAP = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def setup_detection_pipeline(
    dev_info: dai.DeviceInfo,
    visualizer: dai.RemoteConnection,
) -> Optional[Dict[str, Any]]:
    mxid = dev_info.getDeviceId()
    device_instance = None

    print(f"\nAttempting to connect to device for detection: {mxid}...")
    device_instance = dai.Device(dev_info)
    print(f"=== Successfully connected to device: {mxid}")

    cameras_features = device_instance.getConnectedCameraFeatures()

    print(f"    >>> Cameras: {[c.socket.name for c in cameras_features]}")
    print(f"    >>> USB speed: {device_instance.getUsbSpeed().name}")

    pipeline = dai.Pipeline(device_instance)
    print(f"    Pipeline created for device: {mxid}")

    cam_node = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam_node.requestOutput(size=(300, 300), type=dai.ImgFrame.Type.BGR888p)

    model_description = dai.NNModelDescription(
        "mobilenet-ssd", platform=device_instance.getPlatformAsString()
    )
    archive_path = dai.getModelFromZoo(model_description)
    nn_archive_obj = dai.NNArchive(archivePath=archive_path)

    detector_host_node = pipeline.create(ParsingNeuralNetwork).build(
        input=cam_out, nn_source=nn_archive_obj
    )

    annotation_host_node = pipeline.create(AnnotationNode).build(
        input_frame_stream=cam_out,
        input_detections_stream=detector_host_node.out,
        labels=LABEL_MAP,
        outline_color_rgba=generate_vibrant_random_color(),
    )

    visualizer.addTopic(
        topicName=f"Video - {mxid}",
        output=cam_out,
        group=f"{mxid}",
    )
    visualizer.addTopic(
        topicName=f"Annotations - {mxid}",
        output=annotation_host_node.annotation_out,
        group=f"{mxid}",
    )
    print(f"    Pipeline for {mxid} configured. Ready to be started.")
    return {"device": device_instance, "pipeline": pipeline, "mxid": mxid}


def main():
    all_device_infos = dai.Device.getAllAvailableDevices()
    available_devices_info = filter_internal_cameras(all_device_infos)

    if not available_devices_info:
        print("No DepthAI devices found or filtered out.")
        return

    print(
        f"Found {len(available_devices_info)} DepthAI devices to configure for detection."
    )

    visualizer = dai.RemoteConnection(httpPort=HTTP_PORT)

    initialized_setups: List[Dict[str, Any]] = []

    for dev_info in available_devices_info:
        setup_info = setup_detection_pipeline(dev_info, visualizer)
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

    print(
        f"\n{len(active_pipelines_info)} device(s) should be streaming video and detection overlays."
    )

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
            os._exit(0)


if __name__ == "__main__":
    main()
