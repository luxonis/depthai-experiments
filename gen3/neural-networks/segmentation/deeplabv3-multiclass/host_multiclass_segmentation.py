import cv2
import depthai as dai
import numpy as np

NUM_OF_CLASSES = 21

class MulticlassSegmentation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, nn_shape: tuple[int, int]) -> "MulticlassSegmentation":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.nn_shape = nn_shape
        return self

    # preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert (isinstance(preview, dai.ImgFrame))

        frame = preview.getCvFrame()
        first_layer = nn.getAllLayers()[0]
        data = nn.getTensor(first_layer.name).astype(np.int32).flatten().reshape(self.nn_shape)

        found_classes = np.unique(data)
        output_colors = self.decode_deeplabv3p(data)

        frame = cv2.addWeighted(frame,1, output_colors,0.4,0)
        cv2.putText(frame, "Found classes {}".format(found_classes), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 0, 0))
        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def decode_deeplabv3p(self, output_tensor):
        output = output_tensor.reshape(self.nn_shape)

        # Scale to [0 ... 255] and apply colormap
        output = np.array(output) * (255 / NUM_OF_CLASSES)
        output = output.astype(np.uint8)
        output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)

        # Reset the color of 0 class
        output_colors[output == 0] = [0, 0, 0]

        return output_colors
