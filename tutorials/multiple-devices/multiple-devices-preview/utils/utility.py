import depthai as dai
from typing import List, Callable, Dict, Any

import random
import colorsys


def filter_internal_cameras(devices: List[dai.DeviceInfo]) -> List[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def generate_vibrant_random_color() -> tuple[float, float, float, float]:
    hue = random.random()  # Hue (0.0 to 1.0, wraps around)
    saturation = random.uniform(
        0.7, 1.0
    )  # High saturation (0.7 to 1.0 for colorfulness)
    value = random.uniform(0.6, 1.0)  # Medium to high brightness (0.6 to 1.0)

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (r, g, b, 1)

def setup_devices(devices_info: List[dai.DeviceInfo], visualizer: dai.RemoteConnection, setup_device_pipeline: Callable) -> List[Dict[str, Any]]:
    initialized_setups: List[Dict[str, Any]] = []
    for dev_info in devices_info:
        setup_info = setup_device_pipeline(dev_info, visualizer)
        if setup_info:
            initialized_setups.append(setup_info)
        else:
            print(f"--- Failed to set up device {dev_info.getDeviceId()}. Skipping. ---")
    return initialized_setups

def start_pipelines(setups: List[Dict[str, Any]], visualizer: dai.RemoteConnection) -> List[Dict[str, Any]]:
    active_pipelines_info: List[Dict[str, Any]] = []
    for setup in setups:
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
            continue
    return active_pipelines_info

def any_pipeline_running(pipelines: List[Dict[str, Any]]) -> bool:
    """Return True if at least one pipeline is still running."""
    return any(item["pipeline"].isRunning() for item in pipelines)
