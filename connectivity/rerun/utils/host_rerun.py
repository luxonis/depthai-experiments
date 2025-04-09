from typing import Optional

import cv2
import depthai as dai
import numpy as np
import rerun as rr


class Rerun(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()

        self.input_color = self.createInput(
            "color",
            types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)],
            blocking=False,
            queueSize=1,
        )
        self.input_pointcloud = self.createInput(
            "pointcloud",
            types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.PointCloudData, True)],
            blocking=False,
            queueSize=8,
        )
        self.input_left = self.createInput(
            "left",
            types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)],
            blocking=False,
            queueSize=1,
        )
        self.input_right = self.createInput(
            "right",
            types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)],
            blocking=False,
            queueSize=1,
        )
        self._last_color_msg = None

    def build(
        self,
        color: dai.Node.Output,
        left: Optional[dai.Node.Output] = None,
        right: Optional[dai.Node.Output] = None,
        pointcloud: Optional[dai.Node.Output] = None,
    ) -> "Rerun":
        color.link(self.input_color)
        if pointcloud is not None:
            pointcloud.link(self.input_pointcloud)
        if left is not None:
            left.link(self.input_left)
        if right is not None:
            right.link(self.input_right)
        return self

    def run(self) -> None:
        while self.isRunning():
            if self.input_color.has():
                color_msg = self.input_color.get()
                self._last_color_msg = color_msg
                img = cv2.cvtColor(color_msg.getCvFrame(), cv2.COLOR_BGR2RGB)
                rr.log("Color", rr.Image(img))

            if self.input_pointcloud.has():
                pointcloud_msg = self.input_pointcloud.get()
                points = pointcloud_msg.getPoints().astype(np.float64)
                img = cv2.cvtColor(self._last_color_msg.getCvFrame(), cv2.COLOR_BGR2RGB)
                img_pcl = img.reshape(-1, 3)
                # The point cloud comes back reversed, should be fixed in the future
                points[:, 0] = -points[:, 0]
                rr.log("Pointcloud", rr.Points3D(points, colors=img_pcl))

            if self.input_left.has():
                left_msg = self.input_left.get()
                img = cv2.cvtColor(left_msg.getCvFrame(), cv2.COLOR_GRAY2RGB)
                rr.log("Left", rr.Image(img))

            if self.input_right.has():
                right_msg = self.input_right.get()
                img = cv2.cvtColor(right_msg.getCvFrame(), cv2.COLOR_GRAY2RGB)
                rr.log("Right", rr.Image(img))
