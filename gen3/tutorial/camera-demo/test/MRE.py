#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import os
import datetime
from time import sleep

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
frameInterval = 100

def create_videos_from_images():
    dirname = os.path.dirname(__file__)

    ## left video
    videoName1 = os.path.join(dirname, "in_left.mp4")
    leftFilename1 = os.path.join(dirname,"in_left_0.png")
    leftFilename2 = os.path.join(dirname,"in_left_1.png")

    frame1 = cv2.imread(leftFilename1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(leftFilename2, cv2.IMREAD_GRAYSCALE)
    height, width = frame1.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video1 = cv2.VideoWriter(videoName1, fourcc, 1.0, (width, height))

    video1.write(cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
    video1.write(cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR))

    # Release the video write
    video1.release()

    ## right video
    videoName2 = os.path.join(dirname, "in_right.mp4")
    rightFilename1 = os.path.join(dirname,"in_right_0.png")
    rightFilename2 = os.path.join(dirname,"in_right_1.png")

    frame1 = cv2.imread(rightFilename1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(rightFilename2, cv2.IMREAD_GRAYSCALE)
    height, width = frame1.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video2 = cv2.VideoWriter(videoName2, fourcc, 1.0, (width, height))

    video2.write(cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
    video2.write(cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR))

    # Release the video write
    video2.release()



def create_stereo_depth_pipeline_from_dataset(pipeline : dai.Pipeline):
    print("video input -> STEREO -> XLINK OUT")

    dirname = os.path.dirname(__file__)

    in_left_video : dai.node.ReplayVideo = pipeline.create(dai.node.ReplayVideo)
    in_left_video.setReplayVideoFile(os.path.join(dirname, "in_left.mp4"))
    in_left_video.setOutFrameType(dai.ImgFrame.Type.NV12)
    in_left_video.setLoop(True)

    in_right_video : dai.node.ReplayVideo = pipeline.create(dai.node.ReplayVideo)
    in_right_video.setReplayVideoFile(os.path.join(dirname, "in_right.mp4"))
    in_right_video.setOutFrameType(dai.ImgFrame.Type.NV12)
    in_right_video.setLoop(True)

############### CHANGES ###################
    host1 = TestPassthrough(dai.CameraBoardSocket.CAM_B)
    host2 = TestPassthrough(dai.CameraBoardSocket.CAM_C)

    in_left_video.out.link(host1.input)
    in_right_video.out.link(host2.input)

    imageManipLeft = pipeline.create(dai.node.ImageManip)
    host1.output.link(imageManipLeft.inputImage)

    imageManipRight = pipeline.create(dai.node.ImageManip)
    host2.output.link(imageManipRight.inputImage)   

    stereo = pipeline.create(dai.node.StereoDepth).build(imageManipLeft.out, imageManipRight.out)

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    stereo.setInputResolution(1280, 720)

    pipeline.create(DisplayStereo).build(
        monoLeftOut=imageManipLeft.out,
        monoRightOut=imageManipRight.out,
        dispOut=stereo.disparity,
        depthOut=stereo.depth
    )


class TestPassthrough(dai.node.ThreadedHostNode):
    def __init__(self, instanceNum):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()
        self.instanceNum = instanceNum
        self.timestamp = 0

    def run(self):
        while self.isRunning():
            buffer : dai.ImgFrame = self.input.get()

            buffer.setInstanceNum(self.instanceNum)
            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)
            buffer.setTimestamp(tstamp)
            buffer.setType(dai.ImgFrame.Type.RAW8)
            buffer.setWidth(1280)
            buffer.setHeight(720)

            if self.timestamp == 0:
                self.output.send(buffer)
            self.output.send(buffer)

            print('timestamp_ms:', self.timestamp)
            self.timestamp += frameInterval

            


############### CHANGES ###################


class DisplayStereo(dai.node.HostNode):
    def __init__(self):
        self.baseline = 75 #mm
        self.focal = right_intrinsic[0][0]
        self.max_disp = 96
        self.disp_type = np.uint8
        self.disp_levels = 1
        super().__init__()

    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output, dispOut : dai.Node.Output, depthOut : dai.Node.Output) -> "DisplayStereo":
        self.link_args(monoLeftOut, monoRightOut, dispOut, depthOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame, dispFrame : dai.ImgFrame, depthFrame : dai.ImgFrame) -> None:
        cv2.imshow("rectified_left", self.mono_convert_to_cv2_frame(monoLeftFrame))
        cv2.imshow("rectified_right", self.mono_convert_to_cv2_frame(monoRightFrame))
        cv2.imshow("disparity", self.disparity_convert_to_cv2_frame(dispFrame))
        cv2.imshow("depth", self.depth_convert_to_cv2_frame(depthFrame))
        
        if 1: # Optional delay between iterations, host driven pipeline
            sleep(frameInterval / 1000)

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
    with dai.Pipeline() as pipeline:

        create_videos_from_images() 
        create_stereo_depth_pipeline_from_dataset(pipeline)
 
        pipeline.run()

test_pipeline()
