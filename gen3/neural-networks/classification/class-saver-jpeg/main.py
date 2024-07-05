#!/usr/bin/env python3
import csv
from pathlib import Path
import blobconverter
import depthai as dai

from class_saver import ClassSaver

# Start defining a pipeline
with dai.Pipeline() as pipeline:
    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera).build()
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    cam_rgb.preview.link(detection_nn.input)

    # MobilenetSSD label texts
    texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    data_folder_path = Path(__file__).parent / Path('data/')

    for text in texts:
        (data_folder_path / Path(text)).mkdir(parents=True, exist_ok=True)
    (data_folder_path/ Path("raw")).mkdir(parents=True, exist_ok=True)


    with open(data_folder_path / Path("dataset.csv'"), "w") as dataset_file:
        dataset = csv.DictWriter(
            dataset_file,
            ["timestamp", "label", "left", "top", "right", "bottom", "raw_frame", "overlay_frame", "cropped_frame"]
        )
        dataset.writeheader()

        class_saver = pipeline.create(ClassSaver).build(
            rgb=cam_rgb.preview,
            nn_out=detection_nn.out,
            dataset=dataset,
            texts=texts)
        class_saver.set_datafolder_path(data_folder_path)

        # Start pipeline
        pipeline.run()

        if KeyboardInterrupt:
            pass                

        thread = class_saver.get_thread()
        if thread is not None:
            thread.join()
