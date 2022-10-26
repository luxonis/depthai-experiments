import open3d as o3d
from camera import Camera
from typing import List
import numpy as np
import config

class PointCloudVisualizer:
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras
        self.point_cloud = o3d.geometry.PointCloud()

        self.point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
        self.point_cloud_window.register_key_callback(ord('A'), lambda vis: self.align_point_clouds())
        self.point_cloud_window.register_key_callback(ord('D'), lambda vis: self.toggle_depth())
        self.point_cloud_window.register_key_callback(ord('S'), lambda vis: self.save_point_cloud())
        self.point_cloud_window.register_key_callback(ord('R'), lambda vis: self.reset_alignment())
        self.point_cloud_window.register_key_callback(ord('Q'), lambda vis: self.quit())
        self.point_cloud_window.create_window(window_name="Pointcloud")
        self.point_cloud_window.add_geometry(self.point_cloud)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.point_cloud_window.add_geometry(origin)
        view = self.point_cloud_window.get_view_control()
        view.set_constant_z_far(config.max_range*2)

        self.running = True

        while self.running:
            self.update()

    def update(self):
        self.point_cloud.clear()

        for camera in self.cameras:
            camera.update()
            self.point_cloud += camera.point_cloud

        self.point_cloud_window.update_geometry(self.point_cloud)
        self.point_cloud_window.poll_events()
        self.point_cloud_window.update_renderer()

    def align_point_clouds(self):
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]

        master_point_cloud = self.cameras[0].point_cloud
            

        for camera in self.cameras[1:]:
            for iter, radius in zip(max_iter, voxel_radius):
                target_down = master_point_cloud.voxel_down_sample(radius) 
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
                )

                source_down = camera.point_cloud.voxel_down_sample(radius)
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
                )

                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, camera.point_cloud_alignment,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                    )
                )

                camera.point_cloud_alignment = result_icp.transformation

            camera.save_point_cloud_alignment()

    def reset_alignment(self):
        for camera in self.cameras:
            camera.point_cloud_alignment = np.identity(4)
            camera.save_point_cloud_alignment()


    def toggle_depth(self):
        for camera in self.cameras:
            camera.show_depth = not camera.show_depth

    def save_point_cloud(self):
        for camera in self.cameras:
            o3d.io.write_point_cloud(f"sample_data/pcl_{camera.mxid}.ply", camera.point_cloud)

    def quit(self):
        self.running = False