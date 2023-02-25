import numpy as np
import open3d as o3d
import cv2
import random
import config

DISTANCE_THRESHOLD_PLANE = 0.02 # Defines the maximum distance a point can have 
                                # to an estimated plane to be considered an inlier
MAX_ITER_PLANE = 300   # Defines how often a random plane is sampled and verified
N_POINTS_SAMPLED_PLANE = 3 # Defines the number of points that are randomly sampled to estimate a plane


class BoxEstimator():

    def __init__(self):
        self.raw_pcl = o3d.geometry.PointCloud()
        self.plane_pcl = o3d.geometry.PointCloud()
        self.plane_outliers_pcl = o3d.geometry.PointCloud()
        self.box_pcl = o3d.geometry.PointCloud()
        self.box_sides_pcl = o3d.geometry.PointCloud()
        self.top_side_pcl = o3d.geometry.PointCloud()
        self.visualization_pcl = o3d.geometry.PointCloud()

        self.is_calibrated = False
        self.ground_plane_eq = None

        self.height = None
        self.width = None
        self.length = None

        self.bounding_box = None

    def process_pcl(self, raw_pcl):
        self.raw_pcl = raw_pcl
        if len(raw_pcl.points) < 100:
            return 0,0,0 # No box

        if len(self.raw_pcl.points) < 100:
            return 0,0,0 # No box

        self.detect_ground_plane()

        box = self.get_box_pcl() # TODO, check if there is a reasonable box even
        if box is None:
            return 0,0,0 # No box

        self.get_box_top()
        dimensions = self.get_dimensions()
        return dimensions

    
    def detect_ground_plane(self):
        points = o3d.utility.Vector3dVector(np.array([
            [config.point_cloud_range["x_min"], config.point_cloud_range["y_min"], config.point_cloud_range["z_min"]],
            [config.point_cloud_range["x_max"], config.point_cloud_range["y_max"], config.min_box_height]
        ]))
        
        self.plane_pcl = self.raw_pcl.crop(
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        )

        points = o3d.utility.Vector3dVector(np.array([
            [config.point_cloud_range["x_min"], config.point_cloud_range["y_min"], config.min_box_height],
            [config.point_cloud_range["x_max"], config.point_cloud_range["y_max"], config.point_cloud_range["z_max"]]
        ]))
        self.plane_outliers_pcl = self.raw_pcl.crop(
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        )
        
        return self.plane_pcl, self.plane_outliers_pcl
    
    def get_box_pcl(self):
        plane_outliers = self.plane_outliers_pcl

        # Cluster the outliers, the biggest in good conditions should be the box
        # TODO, if the biggest one is not over the threshold -> no box
        labels = np.array(plane_outliers.cluster_dbscan(eps=0.02, min_points=10))

        labels_short = labels[labels != -1]
        if len(labels_short) == 0:
            return None

        y = np.bincount(labels_short).argmax()
        box_indices = np.where(labels == y)[0]
        self.box_pcl = plane_outliers.select_by_index(box_indices)
        return self.box_pcl

    def get_box_top(self):
        points_np = np.asarray(self.box_pcl.points)
        top_plane_eq, top_plane_inliers = self.fit_plane_vec_constraint([0, 0, 1], points_np, 0.01, 30)

        top_plane = self.box_pcl.select_by_index(top_plane_inliers)
        self.top_side_pcl = top_plane
        self.height = abs(top_plane_eq[3])

        points = o3d.utility.Vector3dVector(np.array([
            [config.point_cloud_range["x_min"], config.point_cloud_range["y_min"], config.point_cloud_range["z_min"]],
            [config.point_cloud_range["x_max"], config.point_cloud_range["y_max"], self.height - config.min_box_height]
        ]))
        self.box_sides_pcl = self.box_pcl.crop(
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        )

        return self.height

    def get_dimensions(self):
        upper_plane_points = np.asarray(self.box_sides_pcl.points)
        coordinates = np.c_[upper_plane_points[:, 0], upper_plane_points[:, 1]].astype('float32')
        # np.save("experiments/sample_data/coordinates.npy", coordinates)
        rect = cv2.minAreaRect(coordinates)
        self.bounding_box = cv2.boxPoints(rect)
        self.width, self.length = rect[1][0], rect[1][1]

        return self.length, self.width, self.height

    def fit_plane_vec_constraint(self, norm_vec, pts, thresh=0.05, n_iterations=300):
        best_eq = []
        best_inliers = []

        n_points = pts.shape[0]
        for iter in range(n_iterations):
            id_sample = random.sample(range(0, n_points), 1)
            point = pts[id_sample]
            d = -np.sum(np.multiply(norm_vec, point))
            plane_eq = [*norm_vec, d]
            pt_id_inliers = self.get_plane_inliers(plane_eq, pts, thresh)
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers

        return best_eq, best_inliers
    
    def get_plane_inliers(self, plane_eq, pts, thresh=0.05):
        pt_id_inliers = []
        dist_pt = self.get_pts_distances_plane(plane_eq, pts)

        # Select indexes where distance is bigger than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        return pt_id_inliers

    def get_pts_distances_plane(self, plane_eq, pts):
        dist_pt = (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] 
                + plane_eq[2] * pts[:, 2] + plane_eq[3])\
                / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
        return dist_pt
