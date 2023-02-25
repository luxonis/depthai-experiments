from functools import partial

import cv2

from depthai_sdk import OakCamera


class Trackbar:
    def __init__(self, trackbar_name, window_name, minValue, maxValue, defaultValue, handler):
        cv2.createTrackbar(trackbar_name, window_name, minValue, maxValue, handler)
        cv2.setTrackbarPos(trackbar_name, window_name, defaultValue)


def on_trackbar_change_lambda(visualizer_, value):
    visualizer_.stereo(wls_lambda=value * 100)


def on_trackbar_change_sigma(visualizer_, value):
    visualizer_.stereo(wls_sigma=value / float(10))


with OakCamera() as oak:
    stereo = oak.create_stereo('400p', fps=30)
    stereo.config_wls(wls_level='high')

    visualizer = oak.visualize(stereo.out.disparity)

    window_name = '0_disparity'
    cv2.namedWindow(window_name)
    Trackbar('Lambda', window_name, 0, 255, 80, partial(on_trackbar_change_lambda, visualizer))
    Trackbar('Sigma', window_name, 0, 100, 15, partial(on_trackbar_change_sigma, visualizer))

    oak.start(blocking=True)
