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

    def build(self, preview: dai.Node.Output, disparity: dai.Node.Output, mask: dai.Node.Output, max_disparity: int) -> "DepthSegmentation":
        self.link_args(preview, disparity, mask)

        self.disp_multiplier = 255 / max_disparity
        return self

    def process(self, preview: dai.ImgFrame, disparity: dai.ImgFrame, mask: dai.Buffer) -> None:
        frame = preview.getCvFrame()

        mask_data = mask.getFrame()
        mask_data = cv2.resize(mask_data, frame.shape[:2])

        class_colors = np.asarray([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)
        output_colors = np.take(class_colors, mask_data, axis=0)
        colored_frame = cv2.addWeighted(frame, 1, output_colors, 0.5, 0)

        disp_frame = (disparity.getFrame() * self.disp_multiplier).astype(np.uint8)
        depth_frame = cv2.applyColorMap(disp_frame, JET_CUSTOM)
        cutout_frame = cv2.applyColorMap(disp_frame * mask_data, JET_CUSTOM)

        combined_frame = np.concatenate((colored_frame, cutout_frame, depth_frame), axis=1)

        output = dai.ImgFrame()
        output.setCvFrame(combined_frame, dai.ImgFrame.Type.BGR888p)
        output.setTimestamp(preview.getTimestamp())
        self.output.send(output)
