from collections import deque

import blobconverter
import cv2
import numpy as np
from depthai import NNData

from depthai_sdk import OakCamera

items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>", "<Gansu>",
         "<Guangdong>", "<Guangxi>", "<Guizhou>", "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>", "<HongKong>",
         "<Hubei>", "<Hunan>", "<InnerMongolia>", "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>", "<Macau>",
         "<Ningxia>", "<Qinghai>", "<Shaanxi>", "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>", "<Tianjin>",
         "<Tibet>", "<Xinjiang>", "<Yunnan>", "<Zhejiang>", "<police>", "A", "B", "C", "D", "E", "F", "G", "H", "I",
         "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def decode_recognition(nn_data: NNData):
    rec_data = nn_data.getFirstLayerInt32()
    last_idx = rec_data.index(-1)
    decoded_text = ''.join([items[int(i)] for i in rec_data[:last_idx]])
    return decoded_text


plates = deque(maxlen=10)


def callback(packet):
    frame = packet.frame

    for det in packet.detections:
        bbox = det.top_left + det.bottom_right
        plates.append(cv2.resize(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], (94, 24)))
        cv2.rectangle(packet.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    try:
        for nn_data in packet.nnData:
            print(nn_data)
    except:
        pass

    if len(plates) > 0:
        cv2.imshow('Plates', np.vstack(plates))

    cv2.imshow('Detections', packet.frame)


with OakCamera(replay='chinese_traffic.mp4') as oak:
    color = oak.create_camera('color', resolution='1080p', fps=30)

    det_nn_path = blobconverter.from_zoo(name='vehicle-license-plate-detection-barrier-0106',
                                         shaves=7,
                                         version='2021.4')
    det = oak.create_nn(det_nn_path, color, nn_type='mobilenet')
    det.config_nn(resize_mode='stretch')

    lp_rec_path = blobconverter.from_zoo(name='license-plate-recognition-barrier-0007', shaves=7, version='2021.4')
    lp_rec = oak.create_nn(lp_rec_path, det, decode_fn=decode_recognition)
    lp_rec.config_multistage_nn(labels=[2])  # Enable only license plates

    oak.visualize([lp_rec.out.main], callback=callback, fps=True)
    oak.start(blocking=True)
