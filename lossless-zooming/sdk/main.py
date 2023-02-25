import cv2
import numpy as np

from depthai_sdk import OakCamera

# Constants
ORIGINAL_SIZE = (1920, 1080)  # 4K
SCENE_SIZE = (1280, 720)  # 1080P
AVG_MAX_NUM = 7

x_arr = []
y_arr = []

limits = [SCENE_SIZE[0] // 2, SCENE_SIZE[1] // 2]  # xmin and ymin limits
limits.append(ORIGINAL_SIZE[0] - limits[0])  # xmax limit
limits.append(ORIGINAL_SIZE[1] - limits[1])  # ymax limit


def average_filter(x, y):
    x_arr.append(x)
    y_arr.append(y)
    if AVG_MAX_NUM < len(x_arr): x_arr.pop(0)
    if AVG_MAX_NUM < len(y_arr): y_arr.pop(0)
    x_avg = 0
    y_avg = 0
    for i in range(len(x_arr)):
        x_avg += x_arr[i]
        y_avg += y_arr[i]
    x_avg = x_avg / len(x_arr)
    y_avg = y_avg / len(y_arr)
    if x_avg < limits[0]: x_avg = limits[0]
    if y_avg < limits[1]: y_avg = limits[1]
    if limits[2] < x_avg: x_avg = limits[2]
    if limits[3] < y_avg: y_avg = limits[3]
    return x_avg, y_avg


def callback(packet):
    if len(packet.img_detections.detections) == 0:
        return

    coords = packet.img_detections.detections[0]

    # Get detection center
    x = (coords.xmin + coords.xmax) / 2 * ORIGINAL_SIZE[0]
    y = (coords.ymin + coords.ymax) / 2 * ORIGINAL_SIZE[1] + 100

    x_avg, y_avg = average_filter(x, y)

    bbox = [x_avg - SCENE_SIZE[0] // 2, y_avg - SCENE_SIZE[1] // 2,
            x_avg + SCENE_SIZE[0] // 2, y_avg + SCENE_SIZE[1] // 2]
    bbox = np.int0(bbox)

    roi_frame = packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imshow('Lossless zooming', roi_frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(interleaved=False)

    nn = oak.create_nn('face-detection-retail-0004', color)
    nn.config_nn(conf_threshold=0.7)
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn.out.main, callback=callback)
    oak.start(blocking=True)
