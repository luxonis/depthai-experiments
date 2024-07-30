#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, cam_isp : dai.Node.Output, disparity_out : dai.Node.Output, stereo_cfg : dai.StereoDepthConfig) -> "Display":
        self.stereo_cfg = stereo_cfg
        self.blendedWindowName = "rgb-depth"
        cv2.namedWindow(self.blendedWindowName)
        cv2.createTrackbar('RGB Weight %', self.blendedWindowName, int(rgbWeight*100), 100, self.updateBlendWeights)

        self.link_args(cam_isp, disparity_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, in_frame : dai.ImgFrame, disp_in : dai.ImgFrame) -> None:

        frameRgb = in_frame.getCvFrame()
        frameDisp = disp_in.getFrame()

        maxDisparity = self.stereo_cfg.getMaxDisparity()
        frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)

        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_TURBO)
        frameDisp = np.ascontiguousarray(frameDisp)

        # Need to have both frames in BGR format before blending
        if len(frameDisp.shape) < 3:
            frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
        cv2.imshow(self.blendedWindowName, blended)

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


# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = True
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
device = dai.Device()
with dai.Pipeline(device) as pipeline:

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)

    #Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(fps)
    if downscaleColor: camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    pipeline.create(Display).build(
        cam_isp=camRgb.isp,
        disparity_out=stereo.disparity,
        stereo_cfg=stereo.initialConfig
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
