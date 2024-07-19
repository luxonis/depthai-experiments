#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
# from depthai_sdk import Replay
import argparse
from pathlib import Path
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='depth-people-counting-01', type=str, help="Path to depthai-recording")
args = parser.parse_args()


class Display(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, leftOut, rightOut, depth, disp):
        self.link_args(leftOut, rightOut, depth, disp)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, leftFrame : dai.Buffer, rightFrame : dai.Buffer, depth, disp):
        assert (isinstance(leftFrame, dai.ImgFrame))
        assert (isinstance(rightFrame, dai.ImgFrame))

        cv2.imshow("left", leftFrame.getCvFrame())
        cv2.imshow("right", rightFrame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


frameInterval = 33
class TestPassthrough(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.output = self.createOutput()
        self.timestamp = 0
        self.instanceNum = None


    def run(self):
        while self.isRunning():
            buffer : dai.ImgFrame = self.input.get()

            buffer.setInstanceNum(self.instanceNum)
            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)
            buffer.setTimestamp(tstamp)
            buffer.setTimestampDevice(tstamp)
            buffer.setType(dai.ImgFrame.Type.RAW8)
            
            self.output.send(buffer)
            self.timestamp += frameInterval


    def setInstanceNum(self, instanceNum):
        self.instanceNum = instanceNum


with dai.Pipeline() as pipeline:

    #TODO video input

    path = Path(args.path).resolve().absolute()

    left = pipeline.create(dai.node.ReplayVideo)
    left.setReplayVideoFile(path / 'left.mp4')
    left.setOutFrameType(dai.ImgFrame.Type.RAW8)
    left.setSize(300, 300)

    right = pipeline.create(dai.node.ReplayVideo)
    right.setReplayVideoFile(path / 'right.mp4')
    right.setOutFrameType(dai.ImgFrame.Type.RAW8)
    right.setSize(300, 300)

    # monoLeft = pipeline.create(dai.node.MonoCamera)
    # monoRight = pipeline.create(dai.node.MonoCamera)

    # monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    # monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  
    
    host1 = pipeline.create(TestPassthrough)
    host1.setInstanceNum(dai.CameraBoardSocket.CAM_B)
    host2 = pipeline.create(TestPassthrough)
    host2.setInstanceNum(dai.CameraBoardSocket.CAM_C)

    # monoLeft.out.link(host1.input)
    # monoRight.out.link(host2.input)

    left.out.link(host1.input)
    right.out.link(host2.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=host1.output, right=host2.output)

    pipeline.create(Display).build(
        leftOut=host1.output,
        rightOut=host2.output,
        depth=stereo.depth,
        disp=stereo.disparity
    )

    pipeline.run()
