import json
import os
from pathlib import Path

import cv2
import depthai
import numpy as np
import time
import json


curr_dir = str(Path('.').resolve().absolute())

device = depthai.Device("", False)
pipeline = device.create_pipeline(config={
    'streams': ['meta_d2h', 'color'],
    'ai': {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
    },
    'camera': {'mono': {'resolution_h': 720, 'fps': 30},
               'rgb': {'resolution_h': 1080, 'fps': 30}},
    'app':
    {
        'enable_imu': True
    },

})

while True:
    _, data_list = pipeline.get_available_nnet_and_data_packets(
                        True)
    for packet in data_list:
        if packet.stream_name == "meta_d2h":
            str_ = packet.getDataAsStr()
            dict_ = json.loads(str_)
            if 'imu' in dict_:
                text = 'IMU acc x: {:7.4f}  y:{:7.4f}  z:{:7.4f}'.format(
                    dict_['imu']['accel']['x'], dict_['imu']['accel']['y'], dict_['imu']['accel']['z'])
                print(text)
                text = 'IMU acc-raw x: {:7.4f}  y:{:7.4f}  z:{:7.4f}'.format(
                    dict_['imu']['accelRaw']['x'], dict_['imu']['accelRaw']['y'], dict_['imu']['accelRaw']['z'])
                print(text)