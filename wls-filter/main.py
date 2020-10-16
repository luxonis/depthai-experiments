#!/usr/bin/env python3

from pathlib import Path

import cv2
import depthai

device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["disparity", "rectified_right", ],
    # IGNORE ai for this example. Will be removed later.
    "ai": {
        "blob_file": str(Path('./models/landmarks-regression-retail-0009.blob').resolve().absolute()),
        'camera_input': "right"
    },
    'camera': {
        'mono': {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': 720,
            'fps': 30,
        },
    },
    'app': {
        'sync_video_meta_streams': True,
    },
})

if p is None:
    raise RuntimeError("Error initializing pipelne")


def on_trackbar_change(value):
    device.send_disparity_confidence_threshold(value)
    return


trackbar_name = 'Disparity confidence'
disp_stream = "disparity"
cv2.namedWindow(disp_stream)
conf_thr_slider_min = 0
conf_thr_slider_max = 255
conf_thr_slider_default = 240
cv2.createTrackbar(trackbar_name, disp_stream, conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
cv2.setTrackbarPos(trackbar_name, disp_stream, conf_thr_slider_default)

prev_right = None
prev_disp = None

# lr_check is not supported currently
lr_check = False
wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(lr_check)

_lambda = 8000


def on_trackbar_change_lambda(value):
    global _lambda
    _lambda = value * 100
    return


_sigma = 1.5


def on_trackbar_change_sigma(value):
    global _sigma
    _sigma = value / float(10)
    return


_lambda_trackbar_name = 'Lambda'
wls_stream = "wls_filter"
cv2.namedWindow(wls_stream)
_lambda_slider_min = 0
_lambda_slider_max = 255
_lambda_slider_default = 80
cv2.createTrackbar(_lambda_trackbar_name, wls_stream, _lambda_slider_min, _lambda_slider_max, on_trackbar_change_lambda)
cv2.setTrackbarPos(_lambda_trackbar_name, wls_stream, _lambda_slider_default)

_sigma_trackbar_name = 'Sigma'
_sigma_slider_min = 0
_sigma_slider_max = 100
_sigma_slider_default = 15
cv2.createTrackbar(_sigma_trackbar_name, wls_stream, _sigma_slider_min, _sigma_slider_max, on_trackbar_change_sigma)
cv2.setTrackbarPos(_sigma_trackbar_name, wls_stream, _sigma_slider_default)

while True:
    data_packets = p.get_available_data_packets(blocking=True)

    for packet in data_packets:
        window_name = packet.stream_name
        packetData = packet.getData()
        if packetData is None:
            print('Invalid packet data!')
            continue
        if packet.stream_name == 'rectified_right':
            frame_bgr = packetData
            frame_bgr = cv2.flip(frame_bgr, flipCode=1)
            prev_right = frame_bgr
            cv2.imshow(window_name, frame_bgr)

        if packet.stream_name == 'disparity':
            frame_bgr = packetData
            prev_disp = frame_bgr
            cv2.imshow(window_name, frame_bgr)

    if prev_right is not None:
        if prev_disp is not None:
            # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
            wls_filter.setLambda(_lambda)
            # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
            wls_filter.setSigmaColor(_sigma)
            # print(_lambda)
            # print(_sigma)
            filtered_disp = wls_filter.filter(prev_disp, prev_right)
            cv2.imshow(wls_stream, filtered_disp)

            cv2.normalize(filtered_disp, filtered_disp, 0, 255, cv2.NORM_MINMAX)
            colored_wls = cv2.applyColorMap(filtered_disp, cv2.COLORMAP_JET)
            cv2.imshow(wls_stream + "_color", colored_wls)

            prev_right = None
            prev_disp = None

    if cv2.waitKey(1) == ord('q'):
        break

del device
