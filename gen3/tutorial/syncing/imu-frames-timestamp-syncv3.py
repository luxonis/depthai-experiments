#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from datetime import timedelta
import time


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)
    

# class Display(dai.node.HostNode):
#     def __init__(self) -> None:
#         super().__init__()


#     def build(self, cam_isp : dai.Node.Output, disparity_out : dai.Node.Output, imu_out : dai.Node.Output) -> "Display":
#         self.fps = FPSHandler()
#         self.blendedWindowName = "rgb-depth"
#         cv2.namedWindow(self.blendedWindowName)
#         cv2.createTrackbar('RGB Weight %', self.blendedWindowName, int(rgbWeight*100), 100, self.updateBlendWeights)

#         self.link_args(cam_isp, disparity_out, imu_out)
#         self.sendProcessingToPipeline(True)
#         return self
    

#     def process(self, in_frame : dai.ImgFrame, disp_in : dai.ImgFrame, imu_in) -> None:

#         frameRgb = in_frame.getCvFrame()
#         frameDisp = disp_in.getFrame()

#         self.fps.next_iter()
#         print('FPS', self.fps.fps())

#         frames = {
#             'rgb' : frameRgb,
#             'disp' : frameDisp,
#             'imu' : imu_in
#         }

#         for imu_packet in imu_in.packets:
#             imu_packet : dai.IMUPacket
#             rgb_ts = in_frame.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)
#             disp_ts = disp_in.getTimestampDevice(dai.CameraExposureOffset.MIDDLE)
#             imu_ts = imu_packet.acceleroMeter.getTimestampDevice()
#             self.print_stats(frames, rgb_ts, disp_ts, imu_ts)

#         maxDisparity = 95
#         frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)

#         frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_TURBO)
#         frameDisp = np.ascontiguousarray(frameDisp)

#         # Need to have both frames in BGR format before blending
#         if len(frameDisp.shape) < 3:
#             frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
#         blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
#         cv2.imshow(self.blendedWindowName, blended)

#         if cv2.waitKey(1) == ord('q'):
#             self.stopPipeline()


#     def updateBlendWeights(self, percent_rgb):
#         """
#         Update the rgb and depth weights used to blend depth/rgb image

#         @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
#         """
#         global depthWeight
#         global rgbWeight
#         rgbWeight = float(percent_rgb)/100.0
#         depthWeight = 1.0 - rgbWeight

    
#     def print_stats(self, frames, rgb_ts, stereo_ts, imu_ts):
#         rgb = frames['rgb']
#         disp = frames['disp']
#         imu = frames['imu']

#         print(f"[Seq {rgb.getSequenceNum()}] Mid of RGB exposure ts: {self.td2ms(rgb_ts)}ms, RGB ts: {self.td2ms(rgb.getTimestampDevice())}ms, RGB exposure time: {self.td2ms(rgb.getExposureTime())}ms")
#         print(f"[Seq {disp.getSequenceNum()}] Mid of Stereo exposure ts: {self.td2ms(stereo_ts)}ms, Disparity ts: {self.td2ms(disp.getTimestampDevice())}ms, Stereo exposure time: {self.td2ms(disp.getExposureTime())}ms")
#         print(f"[Seq {imu.acceleroMeter.sequence}] IMU ts: {self.td2ms(imu_ts)}ms")
#         print('-----------')

msgs = dict()
MS_THRESHOLD = 11

def td2ms(td) -> int:
    # Convert timedelta to milliseconds
    return int(td / timedelta(milliseconds=1))

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


def new_msg(msg, name, ts=None):
        synced = add_msg(msg, name, ts)

        if not synced: return

        fps.next_iter()
        print('FPS', fps.fps())
        rgb_ts, rgb = synced['rgb']
        stereo_ts, disp = synced['disp']
        imuTs, imu = synced['imu']
        print(f"[Seq {rgb.getSequenceNum()}] Mid of RGB exposure ts: {td2ms(rgb_ts)}ms, RGB ts: {td2ms(rgb.getTimestampDevice())}ms, RGB exposure time: {td2ms(rgb.getExposureTime())}ms")
        print(f"[Seq {disp.getSequenceNum()}] Mid of Stereo exposure ts: {td2ms(stereo_ts)}ms, Disparity ts: {td2ms(disp.getTimestampDevice())}ms, Stereo exposure time: {td2ms(disp.getExposureTime())}ms")
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

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

device = dai.Device()
with dai.Pipeline(device) as pipeline:

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)
    camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(45)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(45)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    sync_node = pipeline.create(dai.node.Sync)
    sync_node.inputs[]

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 360)
    imu.setBatchReportThreshold(10)
    imu.setMaxBatchReports(10)

    # pipeline.create(Display).build(
    #     cam_isp=camRgb.isp,
    #     disparity_out=stereo.disparity,
    #     imu_out = imu.out
    # )

    # pipeline.run()

    rgb_q = camRgb.isp.createOutputQueue(blocking=False)
    rgb_q.setName('rgb')
    disp_q = stereo.disparity.createOutputQueue(blocking=False)
    disp_q.setName('disp')
    imu_q = imu.out.createOutputQueue(blocking=False)
    imu_q.setName('imu')

    pipeline.start()

    blendedWindowName = "rgb-depth"
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)
    fps = FPSHandler()
    iter_count = 0

    while pipeline.isRunning():
        
        rgb = rgb_q.tryGet()
        disp = disp_q.tryGet()
        imu = imu_q.tryGet()

        # if rgb is not None:
        #     print(f"iteration number {iter_count} has rgb")

        # if disp is not None:
        #     print(f"iteration number {iter_count} has disp")

        # if imu is not None:
        #     print(f"iteration number {iter_count} has imu")

        if imu is not None and rgb is not None and disp is not None:
            print(f"num {iter_count} has all")

        if cv2.waitKey(1) == ord('q'):
            break
        
        iter_count += 1
    