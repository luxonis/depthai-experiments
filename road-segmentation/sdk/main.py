#!/usr/bin/env python3
import blobconverter
import cv2
import numpy as np

from depthai_sdk import toTensorResult, OakCamera


def decode(nn_data):
    data = np.squeeze(toTensorResult(nn_data)["L0317_ReWeight_SoftMax"])
    class_colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    indices = np.argmax(data, axis=0)
    output_colors = np.take(class_colors, indices, axis=0)
    return output_colors


def callback(packet):
    nn_data = packet.img_detections
    output_colors = decode(nn_data)
    cv2.imshow('Road Segmentation', output_colors)


with OakCamera(replay='cars-california-01') as oak:
    color = oak.create_camera('color')

    nn_path = blobconverter.from_zoo(name='road-segmentation-adas-0001', shaves=6)
    nn = oak.create_nn(nn_path, color)

    oak.visualize(nn, callback=callback, fps=True)
    oak.visualize(nn.out.passthrough, fps=True)
    oak.start(blocking=True)
