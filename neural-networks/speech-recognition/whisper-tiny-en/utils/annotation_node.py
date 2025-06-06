import numpy as np
import depthai as dai

from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self.frame_intput = self.createInput()
        self.color_text_input = self.createInput()

        self.frame_out = self.createOutput()
        self.annotaion_out = self.createOutput()

        self.annotations = AnnotationHelper()

    def run(self) -> None:
        while self.isRunning():
            frame_msg = self.frame_intput.get()
            color_msg = self.color_text_input.tryGet()
            if color_msg:
                text = color_msg.text
                color = color_msg.color

                if len(color) == 3:
                    self.annotations = AnnotationHelper()
                    color = np.array(color) / 255.0
                    dai_text_color = dai.Color(color[0], color[1], color[2], 1.0)
                    dai_frame_color = dai.Color(color[0], color[1], color[2], 0.2)

                    self.annotations.draw_text(
                        "Recognized the color: " + text,
                        (0.02, 0.15),
                        size=34,
                        color=dai_text_color,
                        background_color=dai.Color(1, 1, 1, 1),
                    )
                    self.annotations.draw_rectangle(
                        (0.0, 0.0),
                        (1.0, 1.0),
                        fill_color=dai_frame_color,
                        thickness=0.0,
                    )

            self.annotations.draw_text(
                "Press 'r' to record audio",
                (
                    0.002,
                    0.05,
                ),
                size=24,
                color=dai.Color(1, 1, 1, 1),
                background_color=dai.Color(0, 0, 0, 0.5),
            )
            annotations_msg = self.annotations.build(
                timestamp=frame_msg.getTimestamp(),
                sequence_num=frame_msg.getSequenceNum(),
            )

            self.annotaion_out.send(annotations_msg)
            self.frame_out.send(frame_msg)
