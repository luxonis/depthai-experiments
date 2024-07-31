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
    

class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, cam_isp : dai.Node.Output, disparity_out : dai.Node.Output, imu_out : dai.Node.Output) -> "Display":

        self.link_args(cam_isp, imu_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.Buffer, imu : dai.Buffer) -> None:
        assert(isinstance(in_frame, dai.ImgFrame))
        # assert(isinstance(disp_in, dai.ImgFrame))

        frameRgb = in_frame.getCvFrame()

        cv2.imshow("frame", frameRgb)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def updateBlendWeights(self, percent_rgb):
        """
        Update the rgb and depth weights used to blend depth/rgb image

        @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
        """
        global depthWeight
        global rgbWeight
        rgbWeight = float(percent_rgb)/100.0
        depthWeight = 1.0 - rgbWeight

    
    def print_stats(self, frames, rgb_ts, stereo_ts, imu_ts):
        rgb = frames['rgb']
        disp = frames['disp']
        imu = frames['imu']

        print(f"[Seq {rgb.getSequenceNum()}] Mid of RGB exposure ts: {self.td2ms(rgb_ts)}ms, RGB ts: {self.td2ms(rgb.getTimestampDevice())}ms, RGB exposure time: {self.td2ms(rgb.getExposureTime())}ms")
        print(f"[Seq {disp.getSequenceNum()}] Mid of Stereo exposure ts: {self.td2ms(stereo_ts)}ms, Disparity ts: {self.td2ms(disp.getTimestampDevice())}ms, Stereo exposure time: {self.td2ms(disp.getExposureTime())}ms")
        print(f"[Seq {imu.acceleroMeter.sequence}] IMU ts: {self.td2ms(imu_ts)}ms")
        print('-----------')


    def td2ms(self, td) -> int:
        # Convert timedelta to milliseconds
        return int(td / timedelta(milliseconds=1))


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

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 360)
    imu.setBatchReportThreshold(10)
    imu.setMaxBatchReports(10)

    sync_node = pipeline.create(dai.node.Sync)
    camRgb.isp.link(sync_node.inputs['rgb'])
    stereo.disparity.link(sync_node.inputs['disp'])
    imu.out.link(sync_node.inputs['imu'])

    demux = pipeline.create(dai.node.MessageDemux)
    
    sync_node.out.link(demux.input)

    pipeline.create(Display).build(
        cam_isp=demux.outputs['rgb'],
        disparity_out=demux.outputs['disp'],
        imu_out=demux.outputs['imu']
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
    