import depthai as dai
import numpy as np

EDGE_SHAPE = 300

class ReshapeNNOutputEdge(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.shape = (3, EDGE_SHAPE, EDGE_SHAPE)
        
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])


    def build(self, nn_out : dai.Node.Output) -> "ReshapeNNOutputEdge":
        self.link_args(nn_out)
        self.sendProcessingToPipeline(True)
        return self
    
    
    def process(self, nn_frame : dai.NNData) -> None:
        frame = self.get_frame(nn_frame)

        img = dai.ImgFrame()
        img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        
        self.output.send(img)
        
    
    def get_frame(self, imf_frame):
        return np.array(imf_frame.getData()).view(np.float16).reshape(self.shape).transpose(1, 2, 0).astype(np.uint8)