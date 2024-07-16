import cv2
import depthai as dai
import numpy as np

class PersonSegmentation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, nn_shape: tuple[int, int]) -> "PersonSegmentation":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.nn_shape = nn_shape
        return self

    # preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert(isinstance(preview, dai.ImgFrame))

        # If the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, preview.getHeight(), preview.getWidth())
        frame = preview.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        first_layer = nn.getAllLayers()[0]
        data = nn.getTensor(first_layer.name).astype(np.int32).flatten()
        dims = first_layer.dims[::-1]
        data = np.asarray(data, dtype=np.int32).reshape(dims)

        output_colors = self.decode_deeplabv3p(data)
        frame = cv2.addWeighted(frame,1, output_colors,0.2,0)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

    def decode_deeplabv3p(self, output_tensor):
        class_colors = [[0, 0, 0], [0, 255, 0]]
        class_colors = np.asarray(class_colors, dtype=np.uint8)

        output = output_tensor.reshape(self.nn_shape)
        output_colors = np.take(class_colors, output, axis=0)
        return output_colors
