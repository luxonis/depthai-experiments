import open3d as o3d
from box_estimator import BoxEstimator
from camera import Camera
from typing import List
import numpy as np
import config
import cv2

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
        view.set_constant_z_far(config.max_range * 2)

        self.box_estimator = BoxEstimator()

        self.running = True

        while self.running:
            self.update()

    def update(self):
        self.point_cloud.clear()

        for camera in self.cameras:
            camera.update()
            self.point_cloud += camera.point_cloud

        l, w, h = self.box_estimator.process_pcl(self.point_cloud)
        if l*w*h > config.min_box_size:
            print(f"Box size: {l:.2f} x {w:.2f} x {h:.2f}")

        self.vizualise_box()

        for camera in self.cameras:
            if camera.image_frame is not None:
                img = camera.depth_visualization_frame if camera.show_depth else camera.image_frame
                img = self.vizualise_box_2d(img, camera.intrinsics, camera.distortion_coeffs, camera.world_to_cam, camera.point_cloud_alignment)
                cv2.imshow(camera.window_name, img)


    def vizualise_box(self):
        line_set = o3d.geometry.LineSet()
        l, w, h = self.box_estimator.get_dimensions()
        if l*w*h > config.min_box_size:
            # Draw box
            bounding_box = self.box_estimator.bounding_box
            points_floor = np.c_[bounding_box, np.zeros(4)]
            points_top = np.c_[bounding_box, self.box_estimator.height * np.ones(4)]
            box_points = np.concatenate((points_top, points_floor))

            lines = [[0,4], [1,5], [2,6], [3,7], [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4]]

            line_set = o3d.geometry.LineSet(
                points = o3d.utility.Vector3dVector(box_points),
                lines = o3d.utility.Vector2iVector(lines),
            )

            colors = [[1,0,0] for i in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

        self.point_cloud_window.add_geometry(line_set, reset_bounding_box=False)
        self.point_cloud_window.update_geometry(self.point_cloud)
        self.point_cloud_window.poll_events()
        self.point_cloud_window.update_renderer()
        self.point_cloud_window.remove_geometry(line_set, reset_bounding_box=False)


    def vizualise_box_2d(self, img, intrinsic_mat, distortion_coeffs, world_to_cam, point_cloud_alignment=np.identity(4)):
        l, w, h = self.box_estimator.get_dimensions()
        if l*w*h < config.min_box_size:
            return img
        
        points_floor = np.c_[self.box_estimator.bounding_box, np.zeros(4)]
        points_top = np.c_[self.box_estimator.bounding_box, self.box_estimator.height * np.ones(4)]
        box_points = np.concatenate((points_top, points_floor))

        T = np.eye(4)
        T[2,2] = -1
        bbox_pcl = o3d.geometry.PointCloud()
        bbox_pcl.points = o3d.utility.Vector3dVector(box_points)
        bbox_pcl.transform(np.linalg.inv(point_cloud_alignment))
        bbox_pcl.transform(T)
        bbox_pcl.transform(world_to_cam)

        lines = [[0,4], [1,5], [2,6], [3,7], [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4]]

        # object along negative z-axis so need to correct perspective when plotting using OpenCV
        # cord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        box_points = np.array(bbox_pcl.points)
        img_points, _ = cv2.projectPoints(box_points, (0, 0, 0), (0, 0, 0), intrinsic_mat, distortion_coeffs)

        # draw perspective correct point cloud back on the image
        for line in lines:
            p1 = [int(x) for x in img_points[line[0]][0]]
            p2 = [int(x) for x in img_points[line[1]][0]]
            cv2.line(img, p1, p2, (0, 0,255), 2)

        return img


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