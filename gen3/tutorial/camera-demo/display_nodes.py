import cv2
import numpy as np
import depthai as dai


class DisplayMono(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output) -> "DisplayMono":
        self.link_args(monoLeftOut, monoRightOut)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame) -> None:
        cv2.imshow("left", monoLeftFrame.getCvFrame())
        cv2.imshow("right", monoRightFrame.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


class DisplayRGB(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, rgbPreview : dai.Node.Output, rgbVideo : dai.Node.Output) -> "DisplayRGB":
        self.link_args(rgbPreview, rgbVideo)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, rgbPreviewFrame : dai.ImgFrame, rgbVideoFrame : dai.ImgFrame) -> None:
        cv2.imshow("rgb_preview", rgbPreviewFrame.getCvFrame())
        cv2.imshow("rgb_video", rgbVideoFrame.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


class DisplayStereo(dai.node.HostNode):
    def __init__(self):
        super().__init__()


    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output, dispOut : dai.Node.Output) -> "DisplayStereo":
        self.link_args(monoLeftOut, monoRightOut, dispOut)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame, dispFrame : dai.ImgFrame) -> None:

        monoLeftFrame = monoLeftFrame.getCvFrame()
        monoRightFrame = monoRightFrame.getCvFrame()
        dispFrame = cv2.applyColorMap(dispFrame.getCvFrame(), cv2.COLORMAP_HOT)

        cv2.imshow("left", monoLeftFrame)
        cv2.imshow("right", monoRightFrame)
        cv2.imshow("disparity", dispFrame)
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
