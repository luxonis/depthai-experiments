import depthai as dai
import cv2

class DecodeFrameCV2(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        

    def build(self, enc_out) -> "DecodeFrameCV2":
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        return self
    

    def process(self, enc_vid : dai.ImgFrame) -> None:

        frame = cv2.imdecode(enc_vid.getData(), cv2.IMREAD_COLOR)
        
        img = dai.ImgFrame()
        img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        
        self.output.send(img)