import depthai as dai
import av
import time
import cv2

from fractions import Fraction


class Encoder(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        preview: dai.Node.Output,
        stream: dai.Node.Output,
        codec: str,
        output_shape: tuple[int, int],
    ) -> "Encoder":
        self.link_args(preview, stream)
        self.sendProcessingToPipeline(True)

        self.output_container = av.open("video.mp4", "w")
        self.stream = self.output_container.add_stream(codec, rate=30)

        if codec == "mjpeg":
            # We need to set pixel format for MJEPG, for H264/H265 it's yuv420p by default
            self.stream.pix_fmt = "yuvj420p"
        self.stream.time_base = Fraction(1, 1000 * 1000)  # Microseconds
        self.stream.width = output_shape[0]
        self.stream.height = output_shape[1]

        self.start_time = time.time()
        return self

    def process(self, preview: dai.ImgFrame, video_data: dai.ImgFrame) -> None:
        data = video_data.getData()
        # Create new packet with byte array (we can do this because the data is already encoded in MJPEG)
        packet = av.Packet(data)

        # Set frame timestamp
        timestamp = int((time.time() - self.start_time) * 1000 * 1000)
        packet.pts = timestamp
        packet.dts = timestamp
        packet.time_base = self.stream.time_base
        packet.stream = self.stream

        # Mux the packet into container
        self.output_container.mux(packet)

        cv2.imshow("Preview", preview.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            # Flush the encoder
            for packet in self.stream.encode():
                self.output_container.mux(packet)
            self.output_container.close()

            print("Pipeline exited.")
            self.stopPipeline()
