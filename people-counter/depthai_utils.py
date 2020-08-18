import depthai
from pathlib import Path
import consts.resource_paths
import cv2


class DepthAI:
    def __init__(self, model_name, threshold=0.5, fps=None):
        if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
            raise RuntimeError("Error initializing device. Try to reset it.")

        self.p = depthai.create_pipeline(config={
            "streams": ["metaout", "previewout"] if not isinstance(fps, int) else [
                {'name': 'previewout', "max_fps": fps}, {'name': 'metaout', "max_fps": fps}
            ],
            "ai": {
                "blob_file": str(Path(f'./models/{model_name}/model.blob').resolve().absolute()),
                "blob_file_config": str(Path(f'./models/{model_name}/config.json').resolve().absolute())
            }
        })

        if self.p is None:
            raise RuntimeError("Error creating a pipeline!")

        self.entries_prev = []
        self.threshold = threshold
        self.model_name = model_name

    def run(self):
        while True:
            nnet_packets, data_packets = self.p.get_available_nnet_and_data_packets()

            for _, nnet_packet in enumerate(nnet_packets):
                self.entries_prev = []
                for _, e in enumerate(nnet_packet.entries()):
                    if e[0]['image_id'] == -1.0 or e[0]['conf'] == 0.0:
                        break
                    if e[0]['conf'] > self.threshold:
                        if self.model_name == "mobilenet-ssd" and e[0]['label'] != 15:
                            continue
                        self.entries_prev.append(e[0])

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    img_h = frame.shape[0]
                    img_w = frame.shape[1]

                    results = []
                    for e in self.entries_prev:
                        left = int(e['x_min'] * img_w)
                        top = int(e['y_min'] * img_h)
                        right = int(e['x_max'] * img_w)
                        bottom = int(e['y_max'] * img_h)
                        results.append((left, top, right, bottom))

                    yield frame, results

    def __del__(self):
        if hasattr(self, 'p'):
            del self.p
        depthai.deinit_device()
