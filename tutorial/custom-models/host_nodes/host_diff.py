import depthai as dai
import numpy as np
import cv2

DIFF_SHAPE = (720, 720)

class ColorizeDiff(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
    

    def build(self, nnOut : dai.Node.Output) -> "ColorizeDiff":
        self.link_args(nnOut)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, nnData : dai.NNData) -> None:
        frame = self.get_frame(nnData, DIFF_SHAPE)

        img = dai.ImgFrame()
        img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        
        self.output.send(img)


    def get_frame(self, data: dai.NNData, shape):
        diff = data.getFirstTensor().astype(np.float16).flatten().reshape(shape)
        colorize = cv2.normalize(diff, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        return cv2.applyColorMap(colorize, cv2.COLORMAP_JET)