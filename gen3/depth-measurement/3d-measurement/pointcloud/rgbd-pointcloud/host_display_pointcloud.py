import depthai as dai
import cv2
import open3d as o3d
import numpy as np


class DisplayPointCloud(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.visualizer.create_window()
        self.visualizer.register_key_callback(ord('q'), lambda vis: vis.destroy_window())
        self.pcl_data = o3d.geometry.PointCloud()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0., 0., 0.])
        self.visualizer.add_geometry(coordinate_frame)

        self.first_frame = True

    def build(self, preview: dai.Node.Output, pointcloud: dai.Node.Output) -> "DisplayPointCloud":
        self.link_args(preview, pointcloud)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, preview: dai.ImgFrame, pointcloud: dai.PointCloudData) -> None:
        frame = preview.getCvFrame()
        points = pointcloud.getPoints().astype(np.float64)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.pcl_data.points = o3d.utility.Vector3dVector(points)
        colors = (rgb_frame.reshape(-1, 3) / 255.0).astype(np.float64)
        self.pcl_data.colors = o3d.utility.Vector3dVector(colors)

        if self.first_frame:
            self.visualizer.add_geometry(self.pcl_data)
            self.first_frame = False
        else:
            self.visualizer.update_geometry(self.pcl_data)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()
