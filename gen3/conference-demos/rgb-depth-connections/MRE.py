#!/usr/bin/env python3
import math
import cv2
import depthai as dai
import numpy as np
import blobconverter
import datetime

class Display(dai.node.HostNode):
    def __init__(self) -> None:       
        super().__init__()


    def build(self, cam_rgb : dai.Node.Output, depth : dai.Node.Output, nn_out : dai.Node.Output) -> "Display":
        
        self.link_args(cam_rgb, depth) # doesnt sync
        # self.link_args(cam_rgb, nn_out) # syncs
        self.sendProcessingToPipeline(True)
        return self


    def process(self, rgb_frame : dai.ImgFrame, in_depth : dai.ImgFrame) -> None: # this also doesnt sync
        print("ok")
        pass

    # def process(self, rgb_frame : dai.ImgFrame, in_nn : dai.SpatialImgDetections) -> None: # synced
    #     print("ok")
    #     pass


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
            buffer : dai.Buffer = self.input.get()

            tstamp = datetime.timedelta(seconds = self.timestamp // 1000,
                                        milliseconds = self.timestamp % 1000)
            buffer.setTimestamp(tstamp)
            buffer.setTimestampDevice(tstamp)
            
            self.output.send(buffer)
            self.timestamp += frameInterval


# class HostSync:
#     def __init__(self):
#         self.dict = {}

#     def add_msg(self, name, msg):
#         seq = str(msg.getSequenceNum())
#         if seq not in self.dict:
#             self.dict[seq] = {}
#         # print(f"Adding {name} with seq `{seq}`")
#         self.dict[seq][name] = msg

#     def get_msgs(self):
#         remove = []
#         for name in self.dict:
#             remove.append(name)
#             if len(self.dict[name]) == 3:
#                 ret = self.dict[name]
#                 for rm in remove:
#                     del self.dict[rm]
#                 return ret
#         return None


with dai.Pipeline() as pipeline:

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera).build()
    camRgb.initialControl.setManualFocus(130)
    camRgb.setPreviewKeepAspectRatio(False)

    # Properties
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork).build()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    nnPath = blobconverter.from_zoo(name="yolov4_tiny_coco_416x416", zoo_type="depthai", shaves=5)
    spatialDetectionNetwork.setBlobPath(nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.3)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
    spatialDetectionNetwork.setIouThreshold(0.5)

    camRgb.preview.link(spatialDetectionNetwork.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # host1 = pipeline.create(TestPassthrough)
    # host2 = pipeline.create(TestPassthrough)
    # host3 = pipeline.create(TestPassthrough)

    # check how is it done
    # host1.output.link(camRgb.preview)
    # host2.output.link(spatialDetectionNetwork.passthroughDepth)
    # host3.output.link(spatialDetectionNetwork.out)

    pipeline.create(Display).build(
        cam_rgb=camRgb.preview,
        depth=spatialDetectionNetwork.passthroughDepth,
        nn_out=spatialDetectionNetwork.out
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
