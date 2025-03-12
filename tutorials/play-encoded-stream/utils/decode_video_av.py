import av
import depthai as dai


class DecodeVideoAv(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.verbose = False

    def build(self, enc_out, codec) -> "DecodeVideoAv":
        self.codec = av.CodecContext.create(codec, "r")
        self.link_args(enc_out)
        self.inputs["enc_vid"].setMaxSize(1)
        self.inputs["enc_vid"].setBlocking(True)
        return self

    def process(self, enc_vid) -> None:
        data = enc_vid.getData()
        packets = self.codec.parse(data)
        for packet in packets:
            frames = self.codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format="bgr24")

                img = dai.ImgFrame()
                img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
                self.out.send(img)

    @property
    def out(self) -> dai.Node.Output:
        return self._out
