#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import numba as nb
import depthai as dai

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--enable_uvc', default=False, action="store_true",
                    help='Enable UVC output (1080p). Independent from isp/raw streams. '
                         'Needs a sensor resolution of 4K (with ISP 2x downscale), or 1080p')
parser.add_argument('-1', '--res_1080p',  default=False, action="store_true",
                    help='Set sensor res to 1080p, instead of 4K (with UVC) / 12MP')
args = parser.parse_args()

streams = []
# Enable none, one or both streams
streams.append('isp')
streams.append('raw')

''' Packing scheme for RAW10 - MIPI CSI-2
- 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
- stored on 5 bytes (byte0..4) as:
| byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
|    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |
'''
# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

   #for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
if args.res_1080p:
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
elif args.enable_uvc:
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setIspScale(1, 2)
else:
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

# Optional, set manual focus. 255: macro (8cm), about 120..130: infinity
cam.initialControl.setManualFocus(130)
#cam.setFps(20.0)  # Default: 30

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

if 'raw' in streams:
    xout_raw = pipeline.createXLinkOut()
    xout_raw.setStreamName('raw')
    cam.raw.link(xout_raw.input)

if args.enable_uvc:
    uvc = pipeline.createUVC()
    # 'preview' resolution set here to workaround a post-processing limitation
    cam.setPreviewSize(1920, 1080)
    cam.video.link(uvc.input)
    print("UVC 1080p output was enabled.")
    print("Keep this app running, and open another viewer, e.g. guvcview")

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))

capture_flag = False
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        payload = data.getData()
        capture_file_info_str = ('capture_' + name
                                 + '_' + str(width) + 'x' + str(height)
                                 + '_' + str(data.getSequenceNum())
                                )
        if name == 'isp':
            if capture_flag:
                filename = capture_file_info_str + '_P420.yuv'
                print("Saving to file:", filename)
                payload.tofile(filename)
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
        if name == 'raw':
            # Preallocate the output buffer
            unpacked = np.empty(payload.size * 4 // 5, dtype=np.uint16)
            if capture_flag:
                # Save to capture file on bits [9:0] of the 16-bit pixels
                unpack_raw10(payload, unpacked, expand16bit=False)
                filename = capture_file_info_str + '_10bit.bw'
                print("Saving to file:", filename)
                unpacked.tofile(filename)
            # Full range for display, use bits [15:6] of the 16-bit pixels
            unpack_raw10(payload, unpacked, expand16bit=True)
            shape = (height, width)
            bayer = unpacked.reshape(shape).astype(np.uint16)
            # See this for the ordering, at the end of page:
            # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        if capture_flag:  # Save to disk if `c` was pressed
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            bgr = np.ascontiguousarray(bgr)  # just in case
            cv2.imwrite(filename, bgr)
        bgr = np.ascontiguousarray(bgr)  # just in case
        cv2.imshow(name, bgr)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(1)
    if key == ord('c'):
        capture_flag = True
    if key == ord('q'):
        break
