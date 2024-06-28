#!/usr/bin/env python3
import csv
import threading
import time
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np

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

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frame_norm(frame, bbox):
            norm_vals = np.full(len(bbox), frame.shape[0])
            norm_vals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


        def store_data(in_frame, detections):
            timestamp = int(time.time() * 10000)
            raw_frame_path = data_folder_path / f'raw/{timestamp}.jpg'
            cv2.imwrite(raw_frame_path, in_frame)
            for detection in detections:
                debug_frame = in_frame.copy()
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                det_frame = debug_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_path = data_folder_path / f'{texts[detection.label]}/{timestamp}_cropped.jpg'
                cv2.imwrite(cropped_path, det_frame)
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(debug_frame, texts[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                overlay_path = data_folder_path / f'{texts[detection.label]}/{timestamp}_overlay.jpg'
                cv2.imwrite(overlay_path, debug_frame)

                data = {
                    "timestamp": timestamp,
                    "label": texts[detection.label],
                    "left": bbox[0],
                    "top": bbox[1],
                    "right": bbox[2],
                    "bottom": bbox[3],
                    "raw_frame": raw_frame_path,
                    "overlay_frame": overlay_path,
                    "cropped_frame": cropped_path,
                }
                dataset.writerow(data)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_rgb = cam_rgb.preview.createOutputQueue(4, False)
        q_nn = detection_nn.out.createOutputQueue(4, False)

        frame = None
        thread = None
        detections = []
        
        # Start pipeline
        pipeline.start()

        while pipeline.isRunning():
            # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

            if in_nn is not None:
                detections = in_nn.detections

                if frame is not None:
                    thread = threading.Thread(target=store_data, args=(frame, detections))
                    thread.start()

            if frame is not None:
                debug_frame = frame.copy()
                # if the frame is available, draw bounding boxes on it and show the frame
                for detection in detections:
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(debug_frame, texts[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.imshow("rgb", debug_frame)

            if cv2.waitKey(1) == ord('q'):
                pipeline.stop()
                break

        if KeyboardInterrupt:
            pass                

        if thread is not None:
            thread.join()
