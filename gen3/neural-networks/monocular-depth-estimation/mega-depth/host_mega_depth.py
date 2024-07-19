import cv2
import depthai as dai
import numpy as np

class MegaDepth(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, nn_shape: tuple[int, int]) -> "MegaDepth":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        self.nn_shape = nn_shape
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()

        first_layer_name = nn.getAllLayerNames()[0]
        data = nn.getTensor(first_layer_name).flatten().reshape(self.nn_shape)

        # Scale depth to get relative depth
        d_min = np.min(data)
        d_max = np.max(data)
        depth_relative = (data - d_min) / (d_max - d_min)

        depth_relative = np.array(depth_relative) * 255
        depth_relative = depth_relative.astype(np.uint8)
        depth_relative = 255 - depth_relative
        depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

        cv2.imshow("Depth", np.concatenate([frame, depth_relative], axis=1))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
