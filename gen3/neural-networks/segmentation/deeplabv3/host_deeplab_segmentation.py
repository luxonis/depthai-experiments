import cv2
import depthai as dai
import numpy as np

NUM_OF_CLASSES = 21


class DeeplabSegmentation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        preview: dai.Node.Output,
        nn: dai.Node.Output,
        nn_shape: tuple[int, int],
        multiclass: bool,
    ) -> "DeeplabSegmentation":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self.nn_shape = nn_shape
        self.multiclass = multiclass
        return self

    # Preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert isinstance(preview, dai.ImgFrame)

        frame = preview.getCvFrame()
        data = nn.getFirstTensor().astype(np.int32).flatten().reshape(self.nn_shape)
        output_colors = self.decode_deeplabv3p(data)

        if self.multiclass:
            found_classes = np.unique(data)
            cv2.putText(
                frame,
                "Found classes {}".format(found_classes),
                (10, 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                (255, 0, 0),
            )

        frame = cv2.addWeighted(frame, 1, output_colors, 0.3, 0)
        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()

    def decode_deeplabv3p(self, output_tensor):
        if self.multiclass:
            # Scale to [0 ... 255] and apply colormap
            output_tensor = np.array(output_tensor) * (255 / NUM_OF_CLASSES)
            output_tensor = output_tensor.astype(np.uint8)
            output_colors = cv2.applyColorMap(output_tensor, cv2.COLORMAP_JET)

            # Reset the color of 0 class
            output_colors[output_tensor == 0] = [0, 0, 0]
        else:
            class_colors = [[0, 0, 0], [0, 255, 0]]
            class_colors = np.asarray(class_colors, dtype=np.uint8)
            output_colors = np.take(class_colors, output_tensor, axis=0)

        return output_colors
