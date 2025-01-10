import depthai as dai
from rtsp_server import RTSPServer


class StreamOutput(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.server = RTSPServer()

    def build(self, stream: dai.Node.Output) -> "StreamOutput":
        self.link_args(stream)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, stream: dai.ImgFrame) -> None:
        data = stream.getData()
        self.server.send_data(data)
