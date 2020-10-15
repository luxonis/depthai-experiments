import depthai
from pathlib import Path
import cv2


class DepthAI:
    def __init__(self):
        self.device = depthai.Device('', False)

        self.pipeline = self.device.create_pipeline(config={
            "streams": ["metaout", "previewout"],
            "ai": {
                "blob_file": str(Path('./models/person_detection_retail_0013/model.blob').resolve().absolute()),
                "blob_file_config": str(Path('./models/person_detection_retail_0013/config.json').resolve().absolute()),
            }
        })

        if self.pipeline is None:
            raise RuntimeError("Error creating a pipeline!")

        self.network_results = []

    def run(self):
        while True:
            nnet_packets, data_packets = self.pipeline.get_available_nnet_and_data_packets()
            for nnet_packet in nnet_packets:
                self.network_results = [detection.get_dict() for detection in nnet_packet.getDetectedObjects()]

            for packet in data_packets:
                if packet.stream_name == 'previewout':
                    data = packet.getData()
                    data0 = data[0, :, :]
                    data1 = data[1, :, :]
                    data2 = data[2, :, :]
                    frame = cv2.merge([data0, data1, data2])

                    yield frame, self.network_results

    def __del__(self):
        del self.device

for a, b in DepthAI().run():
    print("test")