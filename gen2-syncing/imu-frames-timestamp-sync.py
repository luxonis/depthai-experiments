#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from datetime import timedelta
from depthai_sdk import FPSHandler

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

# Second slowest msg stream is stereo disparity, 45FPS -> ~22ms / 2 -> ~11ms
MS_THRESHOLD = 11

msgs = dict()

def add_msg(msg, name, ts = None):
    if ts is None:
        ts = msg.getTimestamp()

    if not name in msgs:
        msgs[name] = []

    msgs[name].append((ts, msg))

    synced = {}
    for name, arr in msgs.items():
        # Go through all stored messages and calculate the time difference to the target msg.
        # Then sort these msgs to find a msg that's closest to the target time, and check
        # whether it's below 17ms which is considered in-sync.
        diffs = []
        for i, (msg_ts, msg) in enumerate(arr):
            diffs.append(abs(msg_ts - ts))
        if len(diffs) == 0: break
        diffsSorted = diffs.copy()
        diffsSorted.sort()
        dif = diffsSorted[0]

        if dif < timedelta(milliseconds=MS_THRESHOLD):
            # print(f'Found synced {name} with ts {msg_ts}, target ts {ts}, diff {dif}, location {diffs.index(dif)}')
            # print(diffs)
            synced[name] = diffs.index(dif)


    if len(synced) == 3: # We have 3 synced msgs (IMU packet + disp + rgb)
        # print('--------\Synced msgs! Target ts', ts, )
        # Remove older msgs
        for name, i in synced.items():
            msgs[name] = msgs[name][i:]
        ret = {}
        for name, arr in msgs.items():
            ret[name] = arr.pop(0)
            # print(f'{name} msg ts: {ret[name][0]}, diff {abs(ts - ret[name][0]).microseconds / 1000}ms')
        return ret
    return False


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight

# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

def create_pipeline(device):
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)
    camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(45)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(45)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # Linking
    rgbOut = pipeline.create(dai.node.XLinkOut)
    rgbOut.setStreamName("rgb")
    camRgb.isp.link(rgbOut.input)

    disparityOut = pipeline.create(dai.node.XLinkOut)
    disparityOut.setStreamName("disp")
    stereo.disparity.link(disparityOut.input)

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 360)
    imu.setBatchReportThreshold(10)
    imu.setMaxBatchReports(10)

    imuOut = pipeline.create(dai.node.XLinkOut)
    imuOut.setStreamName("imu")
    imu.out.link(imuOut.input)

    return pipeline


def td2ms(td) -> int:
    # Convert timedelta to milliseconds
    return int(td / timedelta(milliseconds=1))

# Connect to device and start pipeline
with dai.Device() as device:
    device.startPipeline(create_pipeline(device))

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)
    fps = FPSHandler()



    def new_msg(msg, name, ts=None):
        synced = add_msg(msg, name, ts)

        if not synced: return

        fps.nextIter()
        print('FPS', fps.fps())
        rgb_ts, rgb = synced['rgb']
        stereo_ts, disp = synced['disp']
        imuTs, imu = synced['imu']
        print(f"[Seq {rgb.getSequenceNum()}] Mid of RGB exposure ts: {td2ms(rgb_ts)}ms, RGB ts: {td2ms(rgb.getTimestampDevice())}ms, RGB exposure time: {td2ms(rgb.getExposureTime(), exposure=True)}ms")
        print(f"[Seq {disp.getSequenceNum()}] Mid of Stereo exposure ts: {td2ms(stereo_ts)}ms, Disparity ts: {td2ms(disp.getTimestampDevice())}ms, Stereo exposure time: {td2ms(disp.getExposureTime(), exposure=True)}ms")
        print(f"[Seq {imu.acceleroMeter.sequence}] IMU ts: {td2ms(imuTs)}ms")
        print('-----------')

        frameRgb = rgb.getCvFrame()

        frameDisp = disp.getFrame()
        maxDisparity = 95
        frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)

        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_TURBO)
        frameDisp = np.ascontiguousarray(frameDisp)

        # Need to have both frames in BGR format before blending
        if len(frameDisp.shape) < 3:
            frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
        cv2.imshow(blendedWindowName, blended)

    while True:
        for name in ['rgb', 'disp', 'imu']:
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                if name == 'imu':
                    for imuPacket in msg.packets:
                        imuPacket: dai.IMUPacket
                        ts = imuPacket.acceleroMeter.getTimestampDevice()
                        new_msg(imuPacket, name, ts)
                else:
                    msg: dai.ImgFrame
                    ts = msg.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)
                    new_msg(msg, name, ts)

        if cv2.waitKey(1) == ord('q'):
            break
