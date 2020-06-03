import logging
import uuid
from pathlib import Path

import consts.resource_paths  # load paths to depthai resources
import cv2
import depthai  # access the camera and its data packets
from imutils.video import FPS

log = logging.getLogger(__name__)


class DepthAI:
    @staticmethod
    def create_pipeline(config):
        if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
            raise RuntimeError("Error initializing device. Try to reset it.")
        log.info("Creating DepthAI pipeline...")

        pipeline = depthai.create_pipeline(config)
        if pipeline is None:
            raise RuntimeError("Pipeline was not created.")
        log.info("Pipeline created.")
        return pipeline

    def __init__(self, model_location, model_label):
        self.model_label = model_label
        self.pipeline = DepthAI.create_pipeline({
            # metaout - contains neural net output
            # previewout - color video
            'streams': ['metaout', 'previewout'],
            "depth": {
                "calibration_file": "",
                "padding_factor": 0.3,
                "depth_limit_m": 10.0
            },
            'ai': {
                "calc_dist_to_bb": True,
                "keep_aspect_ratio": True,
                'blob_file': str(Path(model_location, 'model.blob').absolute()),
                'blob_file_config': str(Path(model_location, 'config.json').absolute())
            },
        })

        self.network_results = []

    def capture(self):
        while True:
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
            for _, nnet_packet in enumerate(nnet_packets):
                self.network_results = []
                # Shape: [1, 1, N, 7], where N is the number of detected bounding boxes
                for _, e in enumerate(nnet_packet.entries()):
                    if e[0]['image_id'] == -1.0 or e[0]['conf'] < 0.5:
                        break

                    self.network_results.append(e[0])

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    if data is None:
                        continue
                    # The format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
                    # change shape (3, 300, 300) -> (300, 300, 3).
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    img_h = frame.shape[0]
                    img_w = frame.shape[1]

                    boxes = []
                    for e in self.network_results:
                        try:
                            boxes.append({
                                'id': uuid.uuid4(),
                                'detector': self.model_label,
                                'conf': e['conf'],
                                'left': int(e['x_min'] * img_w),
                                'top': int(e['y_min'] * img_h),
                                'right': int(e['x_max'] * img_w),
                                'bottom': int(e['y_max'] * img_h),
                                'distance_x': e['distance_x'],
                                'distance_y': e['distance_y'],
                                'distance_z': e['distance_z'],
                            })
                        except:
                            continue
                    print(boxes)
                    yield frame, boxes

    def __del__(self):
        del self.pipeline
        depthai.deinit_device()


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def capture(self):
        for frame, boxes in super().capture():
            self.fps.update()
            for box in boxes:
                cv2.rectangle(frame, (box['left'], box['top']), (box['right'], box['bottom']), (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(box['distance_x'], 1)), (box['left'], box['top'] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "y: {}".format(round(box['distance_y'], 1)), (box['left'], box['top'] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "z: {}".format(round(box['distance_z'], 1)), (box['left'], box['top'] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "conf: {}".format(round(box['conf'], 1)), (box['left'], box['top'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            yield frame, boxes

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
