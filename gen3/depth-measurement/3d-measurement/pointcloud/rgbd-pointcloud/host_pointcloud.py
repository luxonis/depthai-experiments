import cv2
import depthai as dai

from projector_3d import PointCloudVisualizer

class Pointcloud(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, color: dai.Node.Output, left: dai.Node.Output, right: dai.Node.Output,
              depth: dai.Node.Output, cam_intrinsics: list, shape: tuple[int, int]) -> "Pointcloud":
        self.link_args(color, left, right, depth)
        self.sendProcessingToPipeline(True)

        self.pcl_converter = PointCloudVisualizer(cam_intrinsics, shape[0], shape[1])
        return self

    def process(self, color: dai.ImgFrame, left: dai.ImgFrame, right: dai.ImgFrame, depth: dai.ImgFrame) -> None:
        depth_frame = depth.getFrame()
        color_frame = color.getCvFrame()
        left_frame = left.getCvFrame()
        right_frame = right.getCvFrame()

        depth_vis = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_vis = cv2.equalizeHist(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)

        cv2.imshow("Depth", depth_vis)
        cv2.imshow("Color", color_frame)
        cv2.imshow("Rectified_left", left_frame)
        cv2.imshow("Rectified_right", right_frame)

        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        self.pcl_converter.rgbd_to_projection(depth_frame, rgb_frame)
        self.pcl_converter.visualize_pcl()

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
