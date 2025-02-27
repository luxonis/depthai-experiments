import depthai as dai
from .visualizer_utils import xfeat_visualizer


class StereoVersionVisualizer(dai.node.HostNode):
    """
    Custom node for creating visualizations for stereo mode.
    It visualizes both frame and draw lines between matched features and returns the dai.ImgFrame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        left_frame_input: dai.Node.Output,
        right_frame_input: dai.Node.Output,
        tracked_features: dai.Node.Output,
    ) -> "StereoVersionVisualizer":
        self.link_args(left_frame_input, right_frame_input, tracked_features)
        return self

    def process(
        self,
        left_frame: dai.Buffer,
        right_frame: dai.Buffer,
        tracked_features: dai.Buffer,
    ) -> None:
        assert isinstance(left_frame, dai.ImgFrame)
        assert isinstance(right_frame, dai.ImgFrame)
        assert isinstance(tracked_features, dai.TrackedFeatures)

        left_frame = left_frame.getCvFrame()
        right_frame = right_frame.getCvFrame()

        resulting_frame = xfeat_visualizer(
            left_frame,
            right_frame,
            tracked_features.trackedFeatures,
            draw_warp_corners=False,
        )

        output_frame = dai.ImgFrame()
        output_frame.setCvFrame(resulting_frame, dai.ImgFrame.Type.BGR888i)
        self.output.send(output_frame)


class MonoVersionVisualizer(dai.node.HostNode):
    """
    Custom node for creating visualizations for mono mode.
    It visualizes both frame and draw lines between matched features and returns the dai.ImgFrame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.referece_frame = None
        self.set_reference_frame = False
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def setReferenceFrame(self) -> None:
        self.set_reference_frame = True

    def build(
        self, target_frame_input: dai.Node.Output, tracked_features: dai.Node.Output
    ) -> "MonoVersionVisualizer":
        self.link_args(target_frame_input, tracked_features)
        return self

    def process(self, target_frame: dai.Buffer, tracked_features: dai.Buffer) -> None:
        assert isinstance(target_frame, dai.ImgFrame)
        assert isinstance(tracked_features, dai.TrackedFeatures)

        target_frame = target_frame.getCvFrame()

        if self.set_reference_frame:
            self.referece_frame = target_frame
            self.set_reference_frame = False

        if self.referece_frame is not None:
            resulting_frame = xfeat_visualizer(
                self.referece_frame, target_frame, tracked_features.trackedFeatures
            )

        else:
            resulting_frame = target_frame

        output_frame = dai.ImgFrame()
        output_frame.setCvFrame(resulting_frame, dai.ImgFrame.Type.BGR888i)
        self.output.send(output_frame)
