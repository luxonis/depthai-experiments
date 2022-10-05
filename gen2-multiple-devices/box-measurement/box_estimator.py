import numpy as np
import open3d as o3d
import cv2
import copy
import random
import config

DISTANCE_THRESHOLD_PLANE = 0.02 # Defines the maximum distance a point can have 
                                # to an estimated plane to be considered an inlier
MAX_ITER_PLANE = 300   # Defines how often a random plane is sampled and verified
N_POINTS_SAMPLED_PLANE = 3 # Defines the number of points that are randomly sampled to estimate a plane


class BoxEstimator():

    def __init__(self, max_distance):
        self.raw_pcl = o3d.geometry.PointCloud()
        self.plane_pcl = o3d.geometry.PointCloud()
        self.plane_outliers_pcl = o3d.geometry.PointCloud()
        self.box_pcl = o3d.geometry.PointCloud()
        self.top_side_pcl = o3d.geometry.PointCloud()
        self.visualization_pcl = o3d.geometry.PointCloud()

        self.is_calibrated = False
        self.ground_plane_eq = None

        self.height = None
        self.width = None
        self.length = None

        self.bounding_box = None
        self.rotation_matrix = None
        self.translate_vector = None

        self.max_distance = max_distance

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_view_control().set_constant_z_far(config.max_range*2)
        self.vis.add_geometry(self.visualization_pcl)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.vis.add_geometry(origin)

    def vizualise_box(self):
        bounding_box = self.bounding_box
        points_floor = np.c_[bounding_box, np.zeros(4)]
        points_top = np.c_[bounding_box, self.height * np.ones(4)]
        box_points = np.concatenate((points_top, points_floor))

        lines = [[0,4], [1,5], [2,6], [3,7], [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4]]

        line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(box_points),
            lines = o3d.utility.Vector2iVector(lines),
        )

        colors = [[1,0,0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        self.visualization_pcl.points = self.raw_pcl.points
        self.visualization_pcl.colors = self.raw_pcl.colors

        self.vis.add_geometry(line_set, reset_bounding_box=False)
        self.vis.update_geometry(self.visualization_pcl)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.remove_geometry(line_set, reset_bounding_box=False)

    def vizualise_box_2d(self, intrinsic_mat, img):
        bounding_box = self.bounding_box
        points_floor = np.c_[bounding_box, np.zeros(4)]
        points_top = np.c_[bounding_box, self.height * np.ones(4)]
        box_points = np.concatenate((points_top, points_floor))

        # reverse transformations:
        inverse_translation = [-x for x in self.translate_vector]
        inverse_rot_mat = np.linalg.inv(self.rotation_matrix)

        bbox_pcl = o3d.geometry.PointCloud()
        bbox_pcl.points = o3d.utility.Vector3dVector(box_points)
        bbox_pcl.translate(inverse_translation)
        bbox_pcl.rotate(inverse_rot_mat, center=(0,0,0))

        lines = [[0,4], [1,5], [2,6], [3,7], [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4]]
        intrinsic_mat = np.array(intrinsic_mat)

        # object along negative z-axis so need to correct perspective when plotting using OpenCV
        cord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        box_points = np.array(bbox_pcl.points).dot(cord_change_mat.T)
        img_points, _ = cv2.projectPoints(box_points, (0, 0, 0), (0, 0, 0), intrinsic_mat, np.zeros(4, dtype='float32'))

        # draw perspective correct point cloud back on the image
        for line in lines:
            p1 = [int(x) for x in img_points[line[0]][0]]
            p2 = [int(x) for x in img_points[line[1]][0]]
            cv2.line(img, p1, p2, (0, 0,255), 2)

        return img

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
            [config.point_cloud_range["x_max"], config.point_cloud_range["y_max"], 0.01]
        ]))
        
        self.plane_pcl = self.raw_pcl.crop(
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        )

        points = o3d.utility.Vector3dVector(np.array([
            [config.point_cloud_range["x_min"], config.point_cloud_range["y_min"], 0.01],
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
        top_plane_eq, top_plane_inliers = self.fit_plane_vec_constraint([0, 0, 1], points_np, 0.03, 30)

        top_plane = self.box_pcl.select_by_index(top_plane_inliers)
        self.top_side_pcl = top_plane
        self.height = abs(top_plane_eq[3])
        return self.height

    def get_dimensions(self):
        upper_plane_points = np.asarray(self.top_side_pcl.points)
        coordinates = np.c_[upper_plane_points[:, 0], upper_plane_points[:, 1]].astype('float32')
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
