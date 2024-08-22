import cv2
import depthai as dai
import numpy as np

# Custom colormap with 0 mapped to black - better disparity visualization
JET_CUSTOM = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
JET_CUSTOM[0] = [0, 0, 0]

class DepthSegmentation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, disparity: dai.Node.Output
              , nn_shape: tuple[int, int], max_disparity: int) -> "DepthSegmentation":
        self.link_args(preview, nn, disparity)

        self.nn_shape = nn_shape
        self.disp_multiplier = 255 / max_disparity
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData, disparity: dai.ImgFrame) -> None:
        frame = preview.getCvFrame()
        target_shape = (frame.shape[1], frame.shape[0])
        data = nn.getFirstTensor().astype(np.int32).flatten().reshape(self.nn_shape)
        output_colors = cv2.resize(self.decode_deeplabv3p(data), target_shape)

        disp_frame = (disparity.getFrame() * self.disp_multiplier).astype(np.uint8)
        multiplier = self.get_multiplier(data)
        multiplier = cv2.resize(multiplier, target_shape)

        colored_frame = cv2.addWeighted(frame, 1, output_colors, 0.5, 0)
        depth_frame = cv2.applyColorMap(disp_frame, JET_CUSTOM)
        cutout_frame = cv2.applyColorMap(disp_frame * multiplier, JET_CUSTOM)
        # You can add custom code here, for example depth averaging

        combined_frame = np.concatenate((colored_frame, cutout_frame, depth_frame), axis=1)

        output = dai.ImgFrame()
        output.setCvFrame(combined_frame, dai.ImgFrame.Type.BGR888p)
        output.setTimestamp(preview.getTimestamp())
        self.output.send(output)

    def get_multiplier(self, output_tensor):
        class_binary = [[0], [1]]
        class_binary = np.asarray(class_binary, dtype=np.uint8)
        output = output_tensor.reshape(self.nn_shape)
        output_colors = np.take(class_binary, output, axis=0)
        return output_colors

    def decode_deeplabv3p(self, output_tensor):
        class_colors = [[0, 0, 0], [0, 255, 0]]
        class_colors = np.asarray(class_colors, dtype=np.uint8)

        output = output_tensor.reshape(self.nn_shape)
        output_colors = np.take(class_colors, output, axis=0)
        return output_colors
