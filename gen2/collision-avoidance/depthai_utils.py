import logging
import uuid
from pathlib import Path

import cv2
import depthai
from imutils.video import FPS

log = logging.getLogger(__name__)


class DepthAI:
    def create_pipeline(self, config):
        self.device = depthai.Device('', False)
        log.info("Creating DepthAI pipeline...")

        pipeline = self.device.create_pipeline(config)
        if pipeline is None:
            raise RuntimeError("Pipeline was not created.")
        log.info("Pipeline created.")
        return pipeline

    def __init__(self, model_name, threshold):
        self.pipeline = self.create_pipeline({
            # metaout - contains neural net output
            # previewout - color video
            'streams': ['metaout', 'previewout'],
            'ai': {
                "calc_dist_to_bb": True,
                "blob_file": str(Path(f'./models/{model_name}/model.blob').resolve().absolute()),
                "blob_file_config": str(Path(f'./models/{model_name}/config.json').resolve().absolute()),
            },
        })

        self.network_results = []
        self.threshold = threshold
        self.model_name = model_name

    def capture(self):
        while True:
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
            for nnet_packet in nnet_packets:
                self.network_results = list(nnet_packet.getDetectedObjects())

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

                    yield frame, self.network_results

    def __del__(self):
        del self.device


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def capture(self):
        for frame, detections in super().capture():
            self.fps.update()

            img_h = frame.shape[0]
            img_w = frame.shape[1]
            for detection in detections:
                left, top = int(detection.x_min * img_w), int(detection.y_min * img_h)
                right, bottom = int(detection.x_max * img_w), int(detection.y_max * img_h)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "x: {}".format(round(detection.depth_x, 1)), (left, top + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "y: {}".format(round(detection.depth_y, 1)), (left, top + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "z: {}".format(round(detection.depth_z, 1)), (left, top + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "conf: {}".format(round(detection.confidence, 1)), (left, top + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            yield frame, detections

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
