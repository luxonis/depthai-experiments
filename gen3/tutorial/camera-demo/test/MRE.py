#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import os
import datetime
from time import sleep

out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthConfig.MedianFilter.MEDIAN_OFF # TODO

right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
dataset_size = 2
frame_interval = 500


def create_stereo_depth_pipeline_from_dataset(pipeline : dai.Pipeline):
    print("video input -> STEREO -> XLINK OUT")

    host1 = pipeline.create(TestSource)
    host1.set_instance_num(dai.CameraBoardSocket.CAM_B)
    host1.set_name('in_left')
    host2 = pipeline.create(TestSource)
    host2.set_instance_num(dai.CameraBoardSocket.CAM_C)
    host2.set_name('in_right')

    image_manip_left = pipeline.create(dai.node.ImageManip)
    # image_manip_left.initialConfig.setFrameType(dai.ImgFrame.Type.RAW8)
    host1.output.link(image_manip_left.inputImage)

    image_manip_right = pipeline.create(dai.node.ImageManip)
    # image_manip_right.initialConfig.setFrameType(dai.ImgFrame.Type.RAW8)
    host2.output.link(image_manip_right.inputImage)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=image_manip_left.out, right=image_manip_right.out)
    
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    stereo.setInputResolution(1280, 720)

    pipeline.create(DisplayStereo).build(
        monoLeftOut=stereo.syncedLeft,
        monoRightOut=stereo.syncedRight,
        dispOut=stereo.disparity,
        depthOut=stereo.depth
    )


class TestSource(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput()
        self.instance_num = None
        self.timestamp = 0
        self.index = 0
        self.name = None

    def run(self):
        while self.isRunning():
            buffer = dai.ImgFrame()
            path = 'dataset/' + str(self.index) + '/' + self.name + '.png'
            
            if os.path.exists(path):
                print("YES")
            data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)

            buffer.setData(data)
            buffer.setTimestamp(tstamp)
            buffer.setTimestampDevice(tstamp)
            buffer.setInstanceNum(self.instance_num)
            buffer.setType(dai.ImgFrame.Type.RAW8)
            buffer.setWidth(1280)
            buffer.setHeight(720)

            self.output.send(buffer)
            if self.timestamp == 0:
                self.output.send(buffer)

            print("Sent frame: {:25s}".format(path), 'timestamp_ms:', self.timestamp)
            self.timestamp += frame_interval
            self.index = (self.index + 1) % dataset_size

            if 1: # Optional delay between iterations, host driven pipeline
                sleep(frame_interval / 1000)

    def set_instance_num(self, instance_num):
        self.instance_num = instance_num

    def set_name(self, name):
        self.name = name


class DisplayStereo(dai.node.HostNode):
    def __init__(self):
        self.baseline = 75 #mm
        self.focal = right_intrinsic[0][0]
        self.max_disp = 96
        self.disp_type = np.uint8
        self.disp_levels = 1
        if (extended):
            self.max_disp *= 2
        if (subpixel):
            self.max_disp *= 32
            self.disp_type = np.uint16  # 5 bits fractional disparity
            self.disp_levels = 32
        super().__init__()

    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output, dispOut : dai.Node.Output, depthOut : dai.Node.Output) -> "DisplayStereo":
        self.link_args(monoLeftOut, monoRightOut, dispOut, depthOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame, dispFrame : dai.ImgFrame, depthFrame : dai.ImgFrame) -> None:
        
        # cv2.imshow("left", monoLeftFrame.getCvFrame())
        cv2.imshow("rectified_left", self.mono_convert_to_cv2_frame(monoLeftFrame))
        # cv2.imshow("rectified_right", self.mono_convert_to_cv2_frame(monoRightFrame))
        # cv2.imshow("disparity", self.disparity_convert_to_cv2_frame(dispFrame))
        # Skip some streams for now, to reduce CPU load
        # cv2.imshow("depth", self.depth_convert_to_cv2_frame(depthFrame))
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def mono_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        return frame
    
    def depth_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
        return frame
    
    def disparity_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        disp = np.array(data).astype(np.uint8).view(self.disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (self.disp_levels * self.baseline * self.focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / self.max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        return frame




def test_pipeline():
    device = dai.Device()
    with dai.Pipeline(device) as pipeline:

        create_stereo_depth_pipeline_from_dataset(pipeline)
 
        pipeline.run()


test_pipeline()
