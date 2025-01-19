import cv2
import depthai as dai

from box_estimator import BoxEstimator
from projector_3d import PointCloudFromRGBD


class BoxMeasurement(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        color: dai.Node.Output,
        depth: dai.Node.Output,
        cam_intrinsics: list,
        shape: tuple[int, int],
        max_dist: float,
        min_box_size: float,
    ) -> "BoxMeasurement":
        self.link_args(color, depth)
        self.sendProcessingToPipeline(True)

        self.intrinsics = cam_intrinsics
        self.min_box_size = min_box_size

        self.pcl_converter = PointCloudFromRGBD(cam_intrinsics, shape[0], shape[1])
        self.box_estimator = BoxEstimator(max_dist)
        return self

    def process(self, color: dai.ImgFrame, depth: dai.ImgFrame) -> None:
        color_frame = color.getCvFrame()
        depth_frame = depth.getFrame()
        pointcloud = self.pcl_converter.rgbd_to_projection(depth_frame, color_frame)

        l, w, h = self.box_estimator.process_pcl(pointcloud)
        if l * w * h > self.min_box_size:
            self.box_estimator.vizualise_box()

            projection_frame = color_frame.copy()
            self.box_estimator.vizualise_box_2d(self.intrinsics, projection_frame)
            cv2.imshow("Projection", projection_frame)
            print(f"Length in meters: {l:.2f}, Width: {w:.2f}, Height:{h:.2f}")

        cv2.imshow("Preview", color_frame)

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
