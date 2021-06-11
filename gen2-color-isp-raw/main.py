#!/usr/bin/env python3

import cv2
import numpy as np
import numba as nb
import depthai as dai

print('Enter board name:')
board = input()

streams = []
# Enable one or both streams
streams.append('isp')
#streams.append('raw')
streams.append('left')
streams.append('right')
streams.append('disparity')

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
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
#cam.initialControl.setManualFocus(130)
#cam.setImageOrientation(dai.CameraImageOrientation.NORMAL)
#cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

if 'raw' in streams:
    xout_raw = pipeline.createXLinkOut()
    xout_raw.setStreamName('raw')
    cam.raw.link(xout_raw.input)

from_camera = True
out_rectified = False
if from_camera:
    cam_left      = pipeline.createMonoCamera()
    cam_right     = pipeline.createMonoCamera()
else:
    cam_left      = pipeline.createXLinkIn()
    cam_right     = pipeline.createXLinkIn()
stereo            = pipeline.createStereoDepth()
xout_left         = pipeline.createXLinkOut()
xout_right        = pipeline.createXLinkOut()
xout_depth        = pipeline.createXLinkOut()
xout_disparity    = pipeline.createXLinkOut()
xout_rectif_left  = pipeline.createXLinkOut()
xout_rectif_right = pipeline.createXLinkOut()

if from_camera:
    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        #cam.setFps(20.0)
else:
    cam_left .setStreamName('in_left')
    cam_right.setStreamName('in_right')

stereo.setConfidenceThreshold(240)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
#stereo.setMedianFilter(median) # KERNEL_7x7 default
stereo.setLeftRightCheck(1)
stereo.setExtendedDisparity(0)
stereo.setSubpixel(1)
if from_camera:
    # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
    #stereo.loadCalibrationFile(path)
    pass
else:
    stereo.setEmptyCalibration() # Set if the input frames are already rectified
    stereo.setInputResolution(1280, 720)

xout_left        .setStreamName('left')
xout_right       .setStreamName('right')
xout_depth       .setStreamName('depth')
xout_disparity   .setStreamName('disparity')
xout_rectif_left .setStreamName('rectified_left')
xout_rectif_right.setStreamName('rectified_right')

cam_left .out        .link(stereo.left)
cam_right.out        .link(stereo.right)
stereo.syncedLeft    .link(xout_left.input)
stereo.syncedRight   .link(xout_right.input)
#stereo.depth         .link(xout_depth.input)
stereo.disparity     .link(xout_disparity.input)
if out_rectified:
    stereo.rectifiedLeft .link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)


device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (3000, 1500))

capture_flag = False
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        payload = data.getData()
        capture_file_info_str = ('capture_' + board + '_' + name
                                 + '_' + str(width) + 'x' + str(height)
                                 + '_' + str(data.getSequenceNum())
                                )
        if name in ['left', 'right', 'disparity']:
            shape = (height, width)
            if name == 'disparity':
                max_disp = stereo.getMaxDisparity()
                disp = np.array(payload).astype(np.uint8).view(np.uint16)
                frame = (disp * 255. / max_disp).astype(np.uint8)
                bgr = frame.reshape(shape).astype(np.uint8)
            else:
                bgr = payload.reshape(shape).astype(np.uint8)
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
