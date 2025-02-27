import numpy as np
import open3d as o3d


class PointCloudFromRGBD:
    def __init__(self, intrinsic_matrix, width, height):
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(
            np.float64
        )
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic_matrix[0][0],
            intrinsic_matrix[1][1],
            intrinsic_matrix[0][2],
            intrinsic_matrix[1][2],
        )

    def rgbd_to_projection(self, depth_map, rgb, downsample=True):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        if len(rgb.shape) == 3:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
            )
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d
            )

        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.pinhole_camera_intrinsic
            )
        else:
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.pinhole_camera_intrinsic
            )
            if downsample:
                pcl = pcl.voxel_down_sample(voxel_size=0.01)
            # Remove noise
            pcl = pcl.remove_statistical_outlier(30, 0.1)[0]
            self.pcl.points = pcl.points
            self.pcl.colors = pcl.colors
        self.pcl.rotate(
            self.R_camera_to_world, center=np.array([0, 0, 0], dtype=np.float64)
        )
        return self.pcl
