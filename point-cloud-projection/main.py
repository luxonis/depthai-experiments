#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import cv2
import depthai
from projector_3d import PointCloudVisualizer

device = depthai.Device("", False)
pipeline = device.create_pipeline(config={
    'streams': ['right', 'depth'],
    'ai': {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
        "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute()),
    },
    'camera': {'mono': {'resolution_h': 720, 'fps': 30}},
})

if pipeline is None:
    raise RuntimeError("Error creating a pipeline!")

right = None
pcl_converter = None

while True:
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for packet in data_packets:
        if packet.stream_name == "right":
            right = packet.getData()
            cv2.imshow(packet.stream_name, right)
        elif packet.stream_name == "depth":
            frame = packet.getData()
            if right is not None:
                if pcl_converter is None:
                    fd, path = tempfile.mkstemp(suffix='.json')
                    with os.fdopen(fd, 'w') as tmp:
                        json.dump({
                            "width": 1280,
                            "height": 720,
                            "intrinsic_matrix": [item for row in device.get_right_intrinsic() for item in row]
                        }, tmp)
                    pcl_converter = PointCloudVisualizer(path)
                pcd = pcl_converter.rgbd_to_projection(frame, right)
                pcl_converter.visualize_pcd()
            cv2.imshow(packet.stream_name, frame)
    if cv2.waitKey(1) == ord("q"):
        break

if pcl_converter is not None:
    pcl_converter.close_window()
