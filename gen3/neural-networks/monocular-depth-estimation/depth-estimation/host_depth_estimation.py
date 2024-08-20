import cv2
import depthai as dai
import numpy as np


class DepthEstimation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output, nn_shape: tuple[int, int], mbnv2: bool) -> "DepthEstimation":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self.mbnv2 = mbnv2
        self.nn_shape = (nn_shape[1]//2, nn_shape[0]//2) if mbnv2 else nn_shape 
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()
        data = nn.getFirstTensor().flatten().reshape((self.nn_shape[1], self.nn_shape[0]))

        # Scale depth to get relative depth
        d_min = np.min(data)
        d_max = np.max(data)
        depth_relative = (data - d_min) / (d_max - d_min)

        depth_relative = np.array(depth_relative) * 255
        depth_relative = depth_relative.astype(np.uint8)
        depth_relative = 255 - depth_relative
        depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

        if self.mbnv2:
            frame = cv2.resize(frame, self.nn_shape)

        cv2.imshow("Depth", np.concatenate([frame, depth_relative], axis=1))

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

