import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode
from depthai_sdk.callback_context import CallbackContext

NN_WIDTH, NN_HEIGHT = 640, 480


def callback(ctx: CallbackContext):
    packet = ctx.packet

    nn_data = packet.img_detections
    pred = np.array(nn_data.getFirstLayerFp16()).reshape((NN_HEIGHT, NN_WIDTH))

    d_min = np.min(pred)
    d_max = np.max(pred)
    depth_relative = (pred - d_min) / (d_max - d_min)

    # Color it
    depth_relative = np.array(depth_relative) * 255
    depth_relative = depth_relative.astype(np.uint8)
    depth_relative = 255 - depth_relative
    depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

    target_width = int(NN_WIDTH)
    target_height = int(NN_HEIGHT * (NN_WIDTH / NN_HEIGHT) / (16 / 9))

    frame = cv2.resize(packet.frame, (target_width, target_height))
    depth_relative = cv2.resize(depth_relative, (target_width, target_height))

    cv2.imshow('Fast depth', cv2.hconcat([frame, depth_relative]))


with OakCamera() as oak:
    color = oak.create_camera('color')

    fast_depth_path = blobconverter.from_zoo(name='fast_depth_480x640', zoo_type='depthai')
    fast_depth_nn = oak.create_nn(fast_depth_path, color)
    fast_depth_nn.config_nn(aspect_ratio_resize_mode='stretch')

    oak.callback(fast_depth_nn, callback=callback)
    oak.start(blocking=True)
