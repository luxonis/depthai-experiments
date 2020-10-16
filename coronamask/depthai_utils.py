import logging
from pathlib import Path

import cv2
import depthai
from imutils.video import FPS

log = logging.getLogger(__name__)


class DepthAI:
    def create_pipeline(self, config):
        self.device = depthai.Device('', False)
        log.info("Creating DepthAI pipeline...")

        self.pipeline = self.device.create_pipeline(config)
        if self.pipeline is None:
            raise RuntimeError("Pipeline was not created.")
        log.info("Pipeline created.")

    def __init__(self, model_location, model_label):
        self.model_label = model_label
        self.create_pipeline({
            'streams': ['previewout', 'metaout'],
            'ai': {
                'blob_file': str(Path(model_location, 'model.blob').absolute()),
                'blob_file_config': str(Path(model_location, 'config.json').absolute())
            },
        })

        self.network_results = []

    def capture(self):
        while True:
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
            for nnet_packet in nnet_packets:
                self.network_results = list(nnet_packet.getDetectedObjects())

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    # The format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
                    # change shape (3, 300, 300) -> (300, 300, 3).
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    yield frame, self.network_results

    def __del__(self):
        del self.pipeline
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
                color = (0, 255, 0) if detection.label == 2 else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, "Conf: {}".format(detection.confidence), (left, top + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            yield frame, detections

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
