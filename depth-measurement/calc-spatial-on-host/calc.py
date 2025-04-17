import math
import depthai as dai
from utility import TextHelper
from keyboard_reader import KeyboardPress
from measure_distance import SpatialDistance, RegionOfInterest


class HostSpatialsCalc(dai.node.HostNode):
    INITIAL_ROI = RegionOfInterest.from_size(300, 200, 5)

    def __init__(self):
        super().__init__()

        self.text = TextHelper()
        self._roi = self.INITIAL_ROI
        self.step = 3

        self.output_roi = self.createOutput()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        disparity_frames: dai.Node.Output,
        measured_depth: dai.Node.Output,
        keyboard_input: dai.Node.Output,
    ) -> "HostSpatialsCalc":
        self.keyboard_input_q = keyboard_input.createOutputQueue()
        self.link_args(disparity_frames, measured_depth)
        print(
            "\nUse 'w', 'a', 's', 'd' keys to move ROI.\nUse 'r' and 'f' to change ROI size."
        )
        return self

    def process(self, disparity: dai.ImgFrame, depth: dai.Buffer) -> None:
        self._update_roi()
        assert isinstance(depth, SpatialDistance)

        disp = disparity.getCvFrame()
        self.text.rectangle(
            disp, (self._roi.xmin, self._roi.ymin), (self._roi.xmax, self._roi.ymax)
        )

        x = (self._roi.xmin + self._roi.xmax) // 2
        y = (self._roi.ymin + self._roi.ymax) // 2
        self.text.putText(
            disp,
            "X: "
            + (
                "{:.1f}m".format(depth.spatials.x / 1000)
                if not math.isnan(depth.spatials.x)
                else "--"
            ),
            (x + 10, y + 20),
        )
        self.text.putText(
            disp,
            "Y: "
            + (
                "{:.1f}m".format(depth.spatials.y / 1000)
                if not math.isnan(depth.spatials.y)
                else "--"
            ),
            (x + 10, y + 35),
        )
        self.text.putText(
            disp,
            "Z: "
            + (
                "{:.1f}m".format(depth.spatials.z / 1000)
                if not math.isnan(depth.spatials.z)
                else "--"
            ),
            (x + 10, y + 50),
        )

        disparity.setCvFrame(disp, dai.ImgFrame.Type.BGR888i)
        self.output.send(disparity)

    def _update_roi(self) -> None:
        try:
            key_presses: list[KeyboardPress] = self.keyboard_input_q.tryGetAll()
        except dai.MessageQueue.QueueException:
            return
        if key_presses:
            for key_press in key_presses:
                if key_press.key == ord("w"):
                    self._roi.ymin -= self.step
                    self._roi.ymax -= self.step
                elif key_press.key == ord("a"):
                    self._roi.xmin -= self.step
                    self._roi.xmax -= self.step
                elif key_press.key == ord("s"):
                    self._roi.ymin += self.step
                    self._roi.ymax += self.step
                elif key_press.key == ord("d"):
                    self._roi.xmin += self.step
                    self._roi.xmax += self.step
                elif key_press.key == ord("r"):
                    self._increase_roi_size()
                elif key_press.key == ord("f"):
                    self._decrease_roi_size()

            self.output_roi.send(self._roi)

    def _increase_roi_size(self):
        if (self._roi.xmax - self._roi.xmin) < 100:
            self._roi.xmin -= 1
            self._roi.xmax += 1
        if (self._roi.ymax - self._roi.ymin) < 100:
            self._roi.ymin -= 1
            self._roi.ymax += 1

    def _decrease_roi_size(self):
        if (self._roi.xmax - self._roi.xmin) > 6:
            self._roi.xmin += 1
            self._roi.xmax -= 1
        if (self._roi.ymax - self._roi.ymin) > 6:
            self._roi.ymin += 1
            self._roi.ymax -= 1
