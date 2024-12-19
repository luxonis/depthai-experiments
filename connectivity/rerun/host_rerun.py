import depthai as dai
import numpy as np
import rerun as rr
import cv2


class Rerun(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, color: dai.Node.Output, pcl: dai.Node.Output) -> "Rerun":
        self.link_args(color, pcl)
        return self

    def process(self, color: dai.ImgFrame, pcl: dai.PointCloudData) -> None:
        colors = cv2.cvtColor(color.getCvFrame(), cv2.COLOR_BGR2RGB).reshape(-1, 3)

        points = pcl.getPoints().astype(np.float64)
        # The point cloud comes back reversed, should be fixed in the future
        points[:, 0] = -points[:, 0]

        rr.log("Pointcloud", rr.Points3D(points, colors=colors))
