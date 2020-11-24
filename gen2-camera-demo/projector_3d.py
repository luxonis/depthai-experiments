#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import open3d as o3d
import pyvista as pv

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None
        self.count = 1

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb,is_rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
            self.rgbd = rgbd_image
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
            self.rgbd = rgbd_image
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
        # voxel_down_pcd = self.pcl.voxel_down_sample(voxel_size=0.005)
        # pcl, _ = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=0.05)
        # self.pcl.points = pcl.points
        # self.pcl.colors = pcl.colors
        
        return self.pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def save_mesh_from_rgbd(self, path):
        full_path = path + 'rgbd_' + str(self.count) + '.ply'
        volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.integration.TSDFVolumeColorType.Gray32)
        ext = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        print(type(self.rgbd))
        volume.integrate(
                    self.rgbd,
                    self.pinhole_camera_intrinsic,
                    np.linalg.inv(ext))
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        x = o3d.io.write_triangle_mesh(full_path, mesh)
        print(x)
        self.count += 1

    def save_pcd_as_ply(self, path):
        full_path = path + 'pcl_' + str(self.count) + '.ply'
        x = o3d.io.write_point_cloud(full_path, self.pcl)
        self.count += 1
        print(x)

    def save_mesh_as_ply_poisson(self, path):
        full_path = path + 'mesh_poisson_' + str(self.count) + '.ply'
        self.pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        print('has normals')
        print(self.pcl.has_normals())
        print(len(self.pcl.normals))
        # voxel_down_pcd = self.pcl.voxel_down_sample(voxel_size=0.03)
        # voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                             std_ratio=2.0)
        # radii = [0.25, 0.5]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))
        # # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcl)
        # x = o3d.io.write_triangle_mesh(full_path, mesh)

        # voxel_down_pcd = self.pcl.voxel_down_sample(voxel_size=0.02)
        # pcl, _ = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)


        # radii = [0.25, 0.5]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcl, depth=8, width=0, scale=1.1, linear_fit=False)
        x = o3d.io.write_triangle_mesh(full_path, mesh)
           
        # downsampled_point_cloud = np.asarray(voxel_down_pcd.points)
        
        # print(len(downsampled_point_cloud))
        # cloud = pv.PolyData(downsampled_point_cloud).extract_geometry().triangulate()
        # volume = cloud.delaunay_3d(alpha=.3)
        # volume.save(full_path)
        # print(type(cloud))
        # pv.save_meshio(full_path, cloud)

        self.count += 1
        print(self.count)

    def save_mesh_as_ply_ball_rolling(self, path):
        full_path = path + 'mesh_poisson_' + str(self.count) + '.ply'
        self.pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        print('has normals')
        print(self.pcl.has_normals())
        print(len(self.pcl.normals))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        # voxel_down_pcd = self.pcl.voxel_down_sample(voxel_size=0.03)
        # voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                             std_ratio=2.0)
        # radii = [0.25, 0.5]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))
        # # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcl)
        # x = o3d.io.write_triangle_mesh(full_path, mesh)

        # voxel_down_pcd = self.pcl.voxel_down_sample(voxel_size=0.02)
        # pcl, _ = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)


        # radii = [0.25, 0.5]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
        # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcl, depth=8, width=0, scale=1.1, linear_fit=False)
        x = o3d.io.write_triangle_mesh(full_path, mesh)
           
        # downsampled_point_cloud = np.asarray(voxel_down_pcd.points)
        
        # print(len(downsampled_point_cloud))
        # cloud = pv.PolyData(downsampled_point_cloud).extract_geometry().triangulate()
        # volume = cloud.delaunay_3d(alpha=.3)
        # volume.save(full_path)
        # print(type(cloud))
        # pv.save_meshio(full_path, cloud)

        self.count += 1
        print(self.count)


    def save_mesh_as_ply_vista(self, path):
        full_path = path + 'mesh_vista' + str(self.count) + '.ply'
        self.pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        print('has normals')
        print(self.pcl.has_normals())
        print(len(self.pcl.points))
        downsampled_point_cloud = np.asarray(self.pcl.points)
        
        # print(len(downsampled_point_cloud))
        cloud = pv.PolyData(downsampled_point_cloud).extract_geometry().triangulate()
        # volume = cloud.delaunay_3d(alpha=.3)
        # volume.save(full_path)
        # print(type(cloud))
        pv.save_meshio(full_path, cloud)

        self.count += 1
        print(self.count)

    def close_window(self):
        self.vis.destroy_window()
