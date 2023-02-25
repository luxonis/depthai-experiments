from functools import partial

import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera

INPUT_SHAPE = (256, 256)
TARGET_SHAPE = (400, 400)

jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom[0] = [0, 0, 0]


def decode_deeplabv3p(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def get_multiplier(output_tensor):
    class_binary = [[0], [1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors


def crop_to_square(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width - delta]


def callback(disp_multiplier, sync_dict):
    frames = {}

    rgb_packet = sync_dict['0_video']
    disparity_packet = sync_dict['1_disparity;1_out']
    nn_packet = sync_dict['2_out;0_video']

    nn_data = nn_packet.img_detections

    layer1 = nn_data.getFirstLayerInt32()
    # reshape to numpy array
    lay1 = np.asarray(layer1, dtype=np.int32).reshape(*INPUT_SHAPE)
    output_colors = decode_deeplabv3p(lay1)

    # To match depth frames
    output_colors = cv2.resize(output_colors, TARGET_SHAPE)

    frame = rgb_packet.frame
    frame = crop_to_square(frame)
    frame = cv2.resize(frame, TARGET_SHAPE)
    frames['frame'] = frame
    frame = cv2.addWeighted(frame, 1, output_colors, 0.5, 0)
    frames['colored_frame'] = frame

    disparity_frame = disparity_packet.frame
    disparity_frame = (disparity_frame * disp_multiplier).astype(np.uint8)
    disparity_frame = crop_to_square(disparity_frame)
    disparity_frame = cv2.resize(disparity_frame, TARGET_SHAPE)

    # Colorize the disparity
    frames['depth'] = cv2.applyColorMap(disparity_frame, jet_custom)

    multiplier = get_multiplier(lay1)
    multiplier = cv2.resize(multiplier, TARGET_SHAPE)
    depth_overlay = disparity_frame * multiplier
    frames['cutout'] = cv2.applyColorMap(depth_overlay, jet_custom)

    show = np.concatenate((frames['colored_frame'], frames['cutout'], frames['depth']), axis=1)
    cv2.imshow("Combined frame", show)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(interleaved=False)

    stereo = oak.create_stereo('400p')

    disp_multiplier = 255 / stereo.node.initialConfig.getMaxDisparity()

    nn_path = blobconverter.from_zoo(name="deeplab_v3_mnv2_256x256", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(resize_mode='crop')

    oak.sync([color.out.main, stereo.out.disparity, nn.out.main], callback=partial(callback, disp_multiplier))
    oak.start(blocking=True)
