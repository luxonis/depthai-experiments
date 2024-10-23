import time
from concurrent.futures import ThreadPoolExecutor
from csv import DictWriter
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


class ClassSaver(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=10)
        self._data_folder_path = Path("./data")
        self._dict_writer = None
        self._dataset_file = None

    def build(
        self, frames: dai.Node.Output, nn: dai.Node.Output, classes: list[str]
    ) -> "ClassSaver":
        self.link_args(frames, nn)
        self.sendProcessingToPipeline(True)
        self._classes = classes
        return self

    def onStart(self):
        self._init_folders()

    def onStop(self):
        if self._dataset_file is not None:
            self._dataset_file.close()
        self._thread_pool_executor.shutdown()

    def _init_folders(self):
        for i in self._classes:
            (self._data_folder_path / i).mkdir(parents=True, exist_ok=True)
        (self._data_folder_path / "raw").mkdir(parents=True, exist_ok=True)

    def _get_dict_writer(self) -> DictWriter:
        if self._dict_writer is not None:
            return self._dict_writer
        self._dataset_file = open(self._data_folder_path / "dataset.csv", "w")
        self._dict_writer = DictWriter(
            self._dataset_file,
            [
                "timestamp",
                "label",
                "left",
                "top",
                "right",
                "bottom",
                "raw_frame",
                "overlay_frame",
                "cropped_frame",
            ],
        )
        self._dict_writer.writeheader()
        return self._dict_writer

    def process(self, img_frame: dai.ImgFrame, nn_in: dai.ImgDetections):
        self._thread_pool_executor.submit(
            self._store_data, img_frame.getCvFrame(), nn_in.detections
        )

    def _store_data(
        self, in_frame: np.ndarray, detections: list[dai.ImgDetection]
    ) -> None:
        timestamp = int(time.time() * 10000)
        raw_frame_path = self._data_folder_path / f"raw/{timestamp}.jpg"
        cv2.imwrite(raw_frame_path, in_frame)
        for detection in detections:
            debug_frame = in_frame.copy()
            bbox = self._frame_norm(
                in_frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            det_frame = debug_frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            cropped_path = (
                self._data_folder_path
                / f"{self._classes[detection.label]}/{timestamp}_cropped.jpg"
            )
            cv2.imwrite(cropped_path, det_frame)
            cv2.rectangle(
                debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
            )
            cv2.putText(
                debug_frame,
                self._classes[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            overlay_path = (
                self._data_folder_path
                / f"{self._classes[detection.label]}/{timestamp}_overlay.jpg"
            )
            cv2.imwrite(overlay_path, debug_frame)

            data = {
                "timestamp": timestamp,
                "label": self._classes[detection.label],
                "left": bbox[0],
                "top": bbox[1],
                "right": bbox[2],
                "bottom": bbox[3],
                "raw_frame": raw_frame_path,
                "overlay_frame": overlay_path,
                "cropped_frame": cropped_path,
            }
            self._get_dict_writer().writerow(data)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def _frame_norm(self, frame: np.ndarray, bbox: tuple):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
