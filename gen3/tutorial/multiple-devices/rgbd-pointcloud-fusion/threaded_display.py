import threading
import cv2
import numpy as np
import depthai as dai
import open3d as o3d
import config   


class OpencvManager:
    def __init__(self):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        self.keys = []

        self.frames : dict[str, dai.ImgFrame] = {} # window_name -> frame
        self.dx_ids : dict[str, str] = {} # window_name -> device_id

        self.point_cloud : dict[str, o3d.geometry.PointCloud] = {}
        self.point_cloud_window = None

    
    def run(self) -> None:
        while True:
            self.newFrameEvent.wait()
            point_cloud = o3d.geometry.PointCloud()
            for win_name in self.point_cloud.keys():

                           
                self.point_cloud_window.update_geometry(self.point_cloud[win_name])
                self.point_cloud_window.poll_events()
                self.point_cloud_window.update_renderer()

            point_cloud.clear()
                    

    def set_frame(self, point_cloud : o3d.geometry.PointCloud, window_name : str) -> None:
        with self.lock:
            self.point_cloud[window_name] = point_cloud
            self.newFrameEvent.set()


    def set_params(self, point_cloud_window) -> None:
        self.point_cloud_window = point_cloud_window
    

    def set_custom_key(self, key : str, dx_id : str) -> None:
        self.keys.append(key)
        self.dx_ids[key] = dx_id
        self._init_frames()


    def _init_frames(self) -> None:
        for key in self.keys:
            if key not in self.point_cloud.keys():
                self.point_cloud[key] = None


class PointCloudVisualizer:
    def __init__(self):
        self.newFrameEvent = threading.Event()
        self.lock = threading.Lock()
        
        self.frames : dict[str, dai.ImgFrame] = {} # window_name -> frame
        
        self.keys = []
        self.dx_ids : dict[str, str] = {} # window_name -> device_id

        self.point_cloud = o3d.geometry.PointCloud()
        self.pinhole_camera_intrinsic = None
        self.world_to_cam = None
        self.point_cloud_alignment = None

        self.point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
        # self.point_cloud_window.register_key_callback(ord('A'), lambda vis: self.align_point_clouds())
        # self.point_cloud_window.register_key_callback(ord('D'), lambda vis: self.toggle_depth())
        # self.point_cloud_window.register_key_callback(ord('S'), lambda vis: self.save_point_cloud())
        # self.point_cloud_window.register_key_callback(ord('R'), lambda vis: self.reset_alignment())
        # self.point_cloud_window.register_key_callback(ord('Q'), lambda vis: self.quit())
        self.point_cloud_window.create_window(window_name="Pointcloud")
        self.point_cloud_window.add_geometry(self.point_cloud)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.point_cloud_window.add_geometry(origin)
        view = self.point_cloud_window.get_view_control()
        view.set_constant_z_far(config.max_range*2)


    def update(self, point_cloud):
        self.point_cloud.clear()
        self.point_cloud += point_cloud

    
    def set_params(self, pinhole_camera_intrinsic, world_to_cam, point_cloud_alignment) -> None:
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.world_to_cam = world_to_cam
        self.point_cloud_alignment = point_cloud_alignment
    

    def rgbd_to_point_cloud(self, depth_frame, image_frame, downsample=False, remove_noise=False):
        rgb_o3d = o3d.geometry.Image(image_frame)
        df = np.copy(depth_frame).astype(np.float32)
        # df -= 20
        depth_o3d = o3d.geometry.Image(df)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(image_frame.shape) != 3)
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic, self.world_to_cam
        )

        if downsample:
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

        if remove_noise:
            point_cloud = point_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)[0]

        self.point_cloud.points = point_cloud.points
        self.point_cloud.colors = point_cloud.colors

        # correct upside down z axis
        T = np.eye(4)
        T[2,2] = -1
        self.point_cloud.transform(T)

        # apply point cloud alignment transform
        self.point_cloud.transform(self.point_cloud_alignment)

        return self.point_cloud

    # def align_point_clouds(self):
    #     voxel_radius = [0.04, 0.02, 0.01]
    #     max_iter = [50, 30, 14]

    #     master_point_cloud = self.cameras[0].point_cloud
            

    #     for camera in self.cameras[1:]:
    #         for iter, radius in zip(max_iter, voxel_radius):
    #             target_down = master_point_cloud.voxel_down_sample(radius) 
    #             target_down.estimate_normals(
    #                 o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
    #             )

    #             source_down = camera.point_cloud.voxel_down_sample(radius)
    #             source_down.estimate_normals(
    #                 o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
    #             )

    #             result_icp = o3d.pipelines.registration.registration_colored_icp(
    #                 source_down, target_down, radius, camera.point_cloud_alignment,
    #                 o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #                 o3d.pipelines.registration.ICPConvergenceCriteria(
    #                     relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
    #                 )
    #             )

    #             camera.point_cloud_alignment = result_icp.transformation

    #         camera.save_point_cloud_alignment()

    # def reset_alignment(self):
    #     for camera in self.cameras:
    #         camera.point_cloud_alignment = np.identity(4)
    #         camera.save_point_cloud_alignment()


    # def toggle_depth(self):
    #     for camera in self.cameras:
    #         camera.show_depth = not camera.show_depth

    # def save_point_cloud(self):
    #     for camera in self.cameras:
    #         o3d.io.write_point_cloud(f"sample_data/pcl_{camera.mxid}.ply", camera.point_cloud)
