import open3d as o3d
import os

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height, enableViz=True):
        self.depth_map = None
        self.rgb = None
        self.pcl = None
        self.enableViz = enableViz
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        if self.enableViz:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.isstarted = False


    def rgbd_to_projection(self, depth_map, rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity = False)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
            self.pcl.remove_non_finite_points()
        return self.pcl


    def visualize_pcd(self):
        assert self.enableViz , ("enableViz is set False. Set enableViz to True to see point cloud visualization")
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()


    def close_window(self):
        self.vis.destroy_window()

