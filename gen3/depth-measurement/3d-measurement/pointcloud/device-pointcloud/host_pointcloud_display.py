import cv2
import numpy as np
import depthai as dai

from projector_device import PointCloudVisualizer

DOWNSAMPLE_PCL = True  # Downsample the pointcloud before operating on it and visualizing to speed up visualization

class PointcloudDisplay(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.pcl_converter = PointCloudVisualizer()

    def build(self, preview: dai.Node.Output, pointcloud: dai.Node.Output, depth_shape: tuple[int, int]) -> "PointcloudDisplay":
        self.link_args(preview, pointcloud)
        self.sendProcessingToPipeline(True)

        self.shape = depth_shape
        return self

    def process(self, preview: dai.ImgFrame, pointcloud: dai.NNData) -> None:
        pcl_data = pointcloud.getFirstTensor().flatten().reshape(1, 3, self.shape[0], self.shape[1])
        pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
        self.pcl_converter.visualize_pcl(pcl_data, downsample=DOWNSAMPLE_PCL)

        cv2.imshow("Preview", preview.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.pcl_converter.close_window()

            print("Pipeline exited.")
            self.stopPipeline()
