import numpy as np
import open3d as o3d
import cv2
import copy

DISTANCE_THRESHOLD_PLANE = 0.02 # Defines the maximum distance a point can have 
                                # to an estimated plane to be considered an inlier
MAX_ITER_PLANE = 300   # Defines how often a random plane is sampled and verified
N_POINTS_SAMPLED_PLANE = 3 # Defines the number of points that are randomly sampled to estimate a plane


class BoxEstimator():

    def __init__(self):
        self.raw_pcl = None
        self.roi_pcl = None
        self.plane_pcl = None
        self.box_pcl = None
        self.top_side_pcl = None

        self.height = None
        self.width = None
        self.length = None

        self.bounding_box = None
        self.rotation_matrix = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def vizualise_box(self):
        bounding_box = self.bounding_box
        points_floor = np.c_[bounding_box, np.zeros(4)]
        points_top = np.c_[bounding_box, self.height * np.ones(4)]
        box_points = np.concatenate((points_top, points_floor))

        lines = [
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [0,1],
            [1,2],
            [2,3],
            [3,0],
            [4,5],
            [5,6],
            [6,7],
            [7,4],
        ]

        line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(box_points),
            lines = o3d.utility.Vector2iVector(lines),
        )

        colors = [[1,0,0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        if not self.isstarted:
            self.vis.add_geometry(self.raw_pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.add_geometry(line_set, reset_bounding_box=False)
            self.vis.update_geometry(self.raw_pcl)
            self.vis.poll_events()
            self.vis.update_renderer()
            self.vis.remove_geometry(line_set, reset_bounding_box=False)

    def process_pcl(self, raw_pcl):
        self.raw_pcl = raw_pcl
        if len(raw_pcl.points) < 100:
            return 0,0,0
        self.crop_plc()

        if len(self.roi_pcl.points) < 100:
            return 0,0,0
        plane_eq, plane_inliers = self.detect_ground_plane()
        box = self.get_box_pcl(plane_eq, plane_inliers) # TODO, check if there is a reasonable box even
        if box is None:
            return 0,0,0
        self.get_box_top(plane_eq)
        dimensions = self.get_dimensions()
        return dimensions



    def create_rotation_matrix(self, vec_in, vec_target):
        # Create a rotation matrix that rotates vec_in to vec_target
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        v = np.cross(vec_in, vec_target)
        s = np.linalg.norm(v)
        c = np.matmul(vec_in, vec_target)
        v_mat = np.array([[0    ,-v[2], v[1] ],
                        [v[2] , 0   , -v[0]],
                        [-v[1], v[0], 0    ]])
        R = np.identity(3) + v_mat + (np.matmul(v_mat, v_mat) * (1 / (1+c)))
        self.rotation_matrix = R
        return R

    def crop_plc(self):
        # Crop only for the ROI
        raw_pcl = self.raw_pcl
        raw_pcl_np = np.asarray(raw_pcl.points)

        # Calculate point distances
        pcl_dist = np.sqrt(np.sum(np.square(raw_pcl_np), axis=1))

        # Only take points less than 1 meter away
        # TODO, ROI should be done somewhere else
        # (already at RGBD according to user specified bounding box)
        indices = np.nonzero(pcl_dist < 1.5)[0]
        self.roi_pcl = raw_pcl.select_by_index(indices)
        return self.roi_pcl
    
    def detect_ground_plane(self):
        roi_pcl = self.roi_pcl
        # Get the ground plane
        plane_eq, plane_inliers = roi_pcl.segment_plane(
            DISTANCE_THRESHOLD_PLANE, N_POINTS_SAMPLED_PLANE, MAX_ITER_PLANE)

        self.plane_pcl = roi_pcl.select_by_index(plane_inliers)

        return plane_eq, plane_inliers
    
    def get_box_pcl(self, plane_eq, plane_inliers):
        plane_outliers = self.roi_pcl.select_by_index(plane_inliers, invert=True)
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

    def get_box_top(self, plane_eq):
        rot_matrix = self.create_rotation_matrix(plane_eq[0:3], [0,0,1]) #
        self.plane_pcl = self.plane_pcl.rotate(rot_matrix, center=(0,0,0))
        avg_z = np.average(np.asarray(self.plane_pcl.points)[:, 2])

        translate_vector = [0, 0, -avg_z]
        self.plane_pcl = self.plane_pcl.translate(translate_vector)

        self.box_pcl = self.box_pcl.rotate(rot_matrix, center=(0,0,0))
        self.box_pcl = self.box_pcl.translate(translate_vector)

        self.raw_pcl = self.raw_pcl.rotate(rot_matrix, center=(0,0,0))
        self.raw_pcl = self.raw_pcl.translate(translate_vector)

        points_np = np.asarray(self.box_pcl.points)
        zs = points_np[:, 2]
        sorted_zs = np.sort(zs)
        height = np.percentile(sorted_zs, 90)

        upper_plane_points_indices = np.nonzero(zs > 0.8 * height)[0]
        upper_plane = self.box_pcl.select_by_index(upper_plane_points_indices)
        self.top_side_pcl = upper_plane
        self.height = height
        return height

    def get_dimensions(self):
        upper_plane_points = np.asarray(self.top_side_pcl.points)
        coordinates = np.c_[upper_plane_points[:, 0], upper_plane_points[:, 1]].astype('float32')
        rect = cv2.minAreaRect(coordinates)
        self.bounding_box = cv2.boxPoints(rect)
        self.width, self.length = rect[1][0], rect[1][1]

        return self.length, self.width, self.height
