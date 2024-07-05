from csv import DictWriter
from pathlib import Path
import threading
import time
import cv2
import depthai as dai
import numpy as np


class ClassSaver(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._thread = None
        self._data_folder_path = Path("./data")


    def build(self, rgb: dai.Node.Output, nn_out: dai.Node.Output, dataset: DictWriter, texts: list[str]) -> "ClassSaver":
        self.link_args(rgb, nn_out)
        self._texts = texts
        self._dataset = dataset
        return self
    

    def process(self, img_frame: dai.ImgFrame, nn_in: dai.ImgDetections):
        if img_frame is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, img_frame.getHeight(), img_frame.getWidth())
            frame = img_frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if nn_in is not None:
            detections = nn_in.detections

            if frame is not None:
                thread = threading.Thread(target=self._store_data, args=(frame, detections))
                thread.start()

        if frame is not None:
            debug_frame = frame.copy()
            # if the frame is available, draw bounding boxes on it and show the frame
            if detections:
                for detection in detections:
                    bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(debug_frame, self._texts[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("rgb", debug_frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def _store_data(self, in_frame: np.ndarray, detections: list[dai.ImgDetection]) -> None:
        timestamp = int(time.time() * 10000)
        raw_frame_path = self._data_folder_path / f'raw/{timestamp}.jpg'
        cv2.imwrite(raw_frame_path, in_frame)
        for detection in detections:
            debug_frame = in_frame.copy()
            bbox = self._frame_norm(in_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            det_frame = debug_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cropped_path = self._data_folder_path / f'{self._texts[detection.label]}/{timestamp}_cropped.jpg'
            cv2.imwrite(cropped_path, det_frame)
            cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(debug_frame, self._texts[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            overlay_path = self._data_folder_path / f'{self._texts[detection.label]}/{timestamp}_overlay.jpg'
            cv2.imwrite(overlay_path, debug_frame)

            data = {
                "timestamp": timestamp,
                "label": self._texts[detection.label],
                "left": bbox[0],
                "top": bbox[1],
                "right": bbox[2],
                "bottom": bbox[3],
                "raw_frame": raw_frame_path,
                "overlay_frame": overlay_path,
                "cropped_frame": cropped_path,
            }
            self._dataset.writerow(data)


    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def _frame_norm(self, frame: np.ndarray, bbox: tuple):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
    

    def get_thread(self) -> threading.Thread | None:
        return self._thread
    

    def set_datafolder_path(self, path: Path) -> None:
        self._data_folder_path = path