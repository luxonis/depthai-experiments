import cv2
import depthai as dai
import numpy as np
from .box_estimator import BoxEstimator
from .projector_3d import PointCloudFromRGBD


class BoxMeasurement(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.dimensions_output = self.createOutput(
            "dimensions",
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

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

        output_frame = color_frame.copy()
        l, w, h = self.box_estimator.process_pcl(pointcloud)
        
        # Create output frame
        if l * w * h > self.min_box_size:
            self.box_estimator.vizualise_box()
            self.box_estimator.vizualise_box_2d(self.intrinsics, output_frame)
            
            # Add text with dimensions to the output frame
            dimensions_text = f"Length: {l:.2f}m, Width: {w:.2f}m, Height: {h:.2f}m"
            cv2.putText(
                output_frame, 
                dimensions_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            
            # Send dimensions message
            dimensions_msg = dai.Buffer()
            dimensions_msg.setData(f"{l:.2f},{w:.2f},{h:.2f}".encode())
            self.dimensions_output.send(dimensions_msg)

        # Send output frame to visualizer
        out_frame = dai.ImgFrame()
        out_frame.setWidth(output_frame.shape[1])
        out_frame.setHeight(output_frame.shape[0])
        out_frame.setType(dai.ImgFrame.Type.BGR888p)
        out_frame.setData(output_frame.transpose(2, 0, 1).flatten())
        out_frame.setTimestamp(color.getTimestamp())
        self.output.send(out_frame)

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()
