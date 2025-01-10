import depthai as dai
import av
import time


class DecodeVideoAv(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.verbose = False

    def build(self, enc_out, codec) -> "DecodeVideoAv":
        self.codec = av.CodecContext.create(codec, "r")
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        self.inputs["enc_vid"].setMaxSize(1)
        self.inputs["enc_vid"].setBlocking(True)
        return self

    def process(self, enc_vid) -> None:
        data = enc_vid.getData()
        start = time.perf_counter()
        packets = self.codec.parse(data)
        for packet in packets:
            frames = self.codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format="bgr24")
                if self.verbose:
                    print(
                        f"AV decode frame in {(time.perf_counter() - start) * 1000} milliseconds"
                    )

                img = dai.ImgFrame()
                img.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
                self.output.send(img)

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose
