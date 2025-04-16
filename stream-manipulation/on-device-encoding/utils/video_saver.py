from fractions import Fraction
from typing import Tuple
import av
import depthai as dai


class VideoSaver(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def build(
        self,
        encoded_stream: dai.Node.Output,
        codec: str,
        output_shape: Tuple[int, int],
        fps: int,
        output_path: str,
    ) -> "VideoSaver":
        self.link_args(encoded_stream)
        self.output_container = av.open(output_path, "w")

        if codec == "h265":
            self.stream = self.output_container.add_stream("hevc", rate=fps)
        else:
            self.stream = self.output_container.add_stream(codec, rate=fps)

        if codec == "mjpeg":
            # We need to set pixel format for MJPEG, for H264/H265 it's yuv420p by default
            self.stream.pix_fmt = "yuvj420p"

        self.stream.time_base = Fraction(1, 1_000_000)  # Microseconds
        self.stream.width = output_shape[0]
        self.stream.height = output_shape[1]
        self.start_time = None

        return self

    def process(self, encoded_frame: dai.EncodedFrame):
        data = encoded_frame.getData()
        # Create new packet with byte array (we can do this because the data is already encoded)
        packet = av.Packet(data)

        # Set frame timestamp
        if self.start_time is None:
            self.start_time = encoded_frame.getTimestamp()
        timestamp = (
            encoded_frame.getTimestamp() - self.start_time
        ).total_seconds() * 1_000_000

        packet.pts = timestamp
        packet.dts = timestamp
        packet.time_base = self.stream.time_base
        packet.stream = self.stream

        # Mux the packet into container
        self.output_container.mux(packet)
