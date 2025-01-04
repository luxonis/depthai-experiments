import depthai as dai


class YUV2BGR(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.input = self.createInput()
        self.out = self.createOutput()

    def run(self):
        while self.isRunning:
            frame = self.input.get().getCvFrame()
            new_frame = dai.ImgFrame()
            new_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
            self.out.send(new_frame)
