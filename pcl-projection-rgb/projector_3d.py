import open3d as o3d
import os

class PointCloudVisualizer():
    def __init__(self, intrinsic_file, enableViz=True):
        self.depth_map = None
        self.rgb = None
        self.pcl = None
        self.enableViz = enableViz
        assert os.path.isfile(intrinsic_file) , ("Intrisic file not found. Rerun the calibrate.py to generate intrinsic file")
            # print()
        self.pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_file)
        if self.enableViz:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.isstarted = False


    def rgbd_to_projection(self, depth_map, rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
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

