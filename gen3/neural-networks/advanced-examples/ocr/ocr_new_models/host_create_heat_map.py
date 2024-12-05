import depthai as dai

from utils import RotatedRectBuffer
import depthai_nodes
import cv2
import numpy as np

class ProcessDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.passthrough = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output_rect = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)])
        self.output_config = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)])
        self.display = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, frame: dai.Node.Output, detections: dai.Node.Output) -> "ProcessDetections":
        self.link_args(frame, detections)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, detections: dai.Buffer) -> None:
        pass
