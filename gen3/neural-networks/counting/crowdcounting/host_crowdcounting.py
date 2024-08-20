import cv2
import depthai as dai
import numpy as np

class Crowdcounting(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, preview: dai.Node.Output, nn: dai.Node.Output
              , nn_shape: tuple[int, int]) -> "Crowdcounting":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.nn_shape = nn_shape
        return self


    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        # Density map output is 1/8 of input size
        data = nn.getFirstTensor().astype(np.float16).flatten().reshape((30, 52))
        # Predicted count is the sum of the density map
        count = np.sum(data)
        frame = preview.getCvFrame()

        label_count = "Predicted count: {:.2f}".format(count)
        cv2.putText(frame, label_count, (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255))  

        output_colors = self.decode_density_map(data)
        frame = cv2.addWeighted(frame, 1, output_colors, 0.5, 0)
        cv2.imshow("Predict count", frame)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def decode_density_map(self, output_tensor):
        output = np.array(output_tensor) * 255
        output = output.astype(np.uint8)
        output_colors = cv2.applyColorMap(output, cv2.COLORMAP_VIRIDIS)
        output_colors = cv2.resize(output_colors, self.nn_shape, interpolation=cv2.INTER_LINEAR)
        return output_colors