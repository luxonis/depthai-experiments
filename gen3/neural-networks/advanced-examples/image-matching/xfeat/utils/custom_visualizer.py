import depthai as dai
from .visualizer_utils import xfeat_visualizer

class StereoVersionVisualizer(dai.node.ThreadedHostNode):
    """
    Custom node for creating visualizations for stereo mode.
    It visualizes both frame and draw lines between matched features and returns the dai.ImgFrame.
    """
    def __init__(self) -> None:
        super().__init__()
        self.left_frame_input = self.createInput()
        self.right_frame_input = self.createInput()
        self.tracked_features = self.createInput()
        self.out = self.createOutput()

    def run(self):
        while self.isRunning():
            try:
                left_frame: dai.ImgFrame = self.left_frame_input.get()
                right_frame: dai.ImgFrame = self.right_frame_input.get()
                tracked_features: dai.TrackedFeatures = self.tracked_features.get()
            except dai.MessageQueue.QueueException:
                break

            left_frame = left_frame.getCvFrame()
            right_frame = right_frame.getCvFrame()

            resulting_frame = xfeat_visualizer(
                left_frame, right_frame, tracked_features.trackedFeatures, draw_warp_corners=False
            )

            output_frame = dai.ImgFrame()
            output_frame.setCvFrame(resulting_frame, dai.ImgFrame.Type.BGR888i)
            self.out.send(output_frame)


class MonoVersionVisualizer(dai.node.ThreadedHostNode):
    """
    Custom node for creating visualizations for mono mode.
    It visualizes both frame and draw lines between matched features and returns the dai.ImgFrame.
    """
    def __init__(self) -> None:
        super().__init__()
        self.referece_frame = None
        self.set_reference_frame = False
        self.target_frame = self.createInput()
        self.tracked_features = self.createInput()
        self.out = self.createOutput()

    def setReferenceFrame(self) -> None:
        self.set_reference_frame = True

    def run(self):
        while self.isRunning():
            try:
                target_frame: dai.ImgFrame = self.target_frame.get()
                tracked_features: dai.TrackedFeatures = self.tracked_features.get()
            except dai.MessageQueue.QueueException:
                break

            target_frame = target_frame.getCvFrame()

            if self.set_reference_frame:
                self.referece_frame = target_frame
                self.set_reference_frame = False

            if self.referece_frame is not None:
                resulting_frame = xfeat_visualizer(self.referece_frame, target_frame, tracked_features.trackedFeatures)

            else:
                resulting_frame = target_frame

            output_frame = dai.ImgFrame()
            output_frame.setCvFrame(resulting_frame, dai.ImgFrame.Type.BGR888i)
            self.out.send(output_frame)