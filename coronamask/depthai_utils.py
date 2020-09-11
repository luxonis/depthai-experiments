import json
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
            for _, nnet_packet in enumerate(nnet_packets):
                self.network_results = []
                for _, e in enumerate(nnet_packet.entries()):
                    if e[0]['image_id'] == -1.0 or e[0]['conf'] == 0.0:
                        break

                    if e[0]['conf'] > 0.5 and e[0]['label'].is_integer():
                        self.network_results.append(e[0])

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
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
                                # 'id': uuid.uuid4(),
                                'label': e['label'],
                                'detector': self.model_label,
                                'conf': e['conf'],
                                'left': int(e['x_min'] * img_w),
                                'top': int(e['y_min'] * img_h),
                                'right': int(e['x_max'] * img_w),
                                'bottom': int(e['y_max'] * img_h),
                            })
                        except:
                            continue
                    yield frame, boxes

    def __del__(self):
        del self.pipeline
        del self.device


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def capture(self):
        for frame, boxes in super().capture():
            self.fps.update()
            for box in boxes:
                color = (0, 255, 0) if box['label'] == 2 else (0, 0, 255)
                cv2.rectangle(frame, (box['left'], box['top']), (box['right'], box['bottom']), color, 2)
                cv2.putText(frame, "Conf: {}".format(box['conf']), (box['left'], box['top'] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            yield frame, boxes

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
