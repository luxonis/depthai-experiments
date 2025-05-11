import numpy as np
import cv2
import random
import depthai as dai
from .img_annotation_helper import AnnotationHelper
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

DISTANCE_THRESHOLD_PLANE = 0.02  # Defines the maximum distance a point can have to an estimated plane to be considered an inlier
MAX_ITER_PLANE_RANSAC = 300      # Defines how often a random plane is sampled and verified for RANSAC
N_POINTS_SAMPLED_PLANE_RANSAC = (
    3  # Defines the number of points that are randomly sampled to estimate a plane for RANSAC
)


def _create_empty_pcl():
    """Creates an empty point cloud structure."""
    return {'points': np.empty((0, 3), dtype=np.float64), 
            'colors': np.empty((0, 3), dtype=np.float64)}

def _pcl_is_empty(pcl_dict):
    """Checks if the point cloud dictionary is empty."""
    return pcl_dict['points'].shape[0] == 0

def _pcl_len(pcl_dict):
    """Returns the number of points in the point cloud dictionary."""
    return pcl_dict['points'].shape[0]

def _rotate_pcl(pcl_dict, rotation_matrix, center=np.array([0.0, 0.0, 0.0])):
    """Rotates points in the point cloud dictionary."""
    if _pcl_is_empty(pcl_dict):
        return pcl_dict.copy()
    points_centered = pcl_dict['points'] - center
    rotated_points = (rotation_matrix @ points_centered.T).T + center
    # Ensure colors are copied if they exist
    copied_colors = pcl_dict['colors'].copy() if pcl_dict['colors'] is not None and pcl_dict['colors'].size > 0 else np.empty((0,3), dtype=np.float64)
    return {'points': rotated_points, 'colors': copied_colors}

def _translate_pcl(pcl_dict, translation_vector):
    """Translates points in the point cloud dictionary."""
    if _pcl_is_empty(pcl_dict):
        return pcl_dict.copy()
    translated_points = pcl_dict['points'] + translation_vector
    copied_colors = pcl_dict['colors'].copy() if pcl_dict['colors'] is not None and pcl_dict['colors'].size > 0 else np.empty((0,3), dtype=np.float64)
    return {'points': translated_points, 'colors': copied_colors}

def _select_by_index(pcl_dict, indices, invert=False):
    """Selects points and colors by indices from the point cloud dictionary."""
    if _pcl_is_empty(pcl_dict): # Primary check for empty point cloud
        return _create_empty_pcl()

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    num_total_points = _pcl_len(pcl_dict)

    if invert:
        if indices.size == 0: # Invert selection with empty indices means select all
            selected_indices = np.arange(num_total_points)
        else:
            mask = np.ones(num_total_points, dtype=bool)
            # Ensure indices are within bounds before using them for masking
            valid_indices_for_mask = indices[(indices >= 0) & (indices < num_total_points)]
            if valid_indices_for_mask.size > 0:
                 mask[valid_indices_for_mask] = False
            selected_indices = np.where(mask)[0]
    else:
        if indices.size == 0: # Select empty if indices are empty and not inverting
            return _create_empty_pcl()
        # Ensure indices are within bounds
        selected_indices = indices[(indices >= 0) & (indices < num_total_points)]

    if selected_indices.size == 0: # If after all logic, no indices are selected
        return _create_empty_pcl()
        
    selected_points = pcl_dict['points'][selected_indices]
    
    has_colors = pcl_dict['colors'] is not None and pcl_dict['colors'].shape[0] == num_total_points
    selected_colors = pcl_dict['colors'][selected_indices] if has_colors else np.empty((selected_points.shape[0],3), dtype=np.float64)
    if not has_colors or selected_colors.shape[0] != selected_points.shape[0]: # Fallback if color selection failed
        selected_colors = np.empty((selected_points.shape[0],3), dtype=np.float64)


    return {'points': selected_points, 'colors': selected_colors}


def _voxel_down_sample(pcl_dict, voxel_size):
    """Performs voxel downsampling on the point cloud."""
    if _pcl_is_empty(pcl_dict) or voxel_size <= 0:
        return pcl_dict.copy()

    points = pcl_dict['points']
    # Ensure points is not empty before proceeding
    if points.shape[0] == 0:
        return pcl_dict.copy()
        
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # np.unique returns unique rows and the first index of occurrence
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    return _select_by_index(pcl_dict, unique_indices)


def _remove_statistical_outlier(pcl_dict, nb_neighbors, std_ratio):
    """Removes statistical outliers from the point cloud."""
    if _pcl_is_empty(pcl_dict) or _pcl_len(pcl_dict) <= nb_neighbors :
        return pcl_dict.copy(), np.array([], dtype=int)

    points = pcl_dict['points']
    # This check is technically covered by _pcl_is_empty or _pcl_len, but good for clarity
    if points.shape[0] == 0:
        return _create_empty_pcl(), np.array([], dtype=int)

    # Ensure enough points for NearestNeighbors
    if points.shape[0] <= nb_neighbors:
        return pcl_dict.copy(), np.arange(points.shape[0]) # Or consider all as inliers if too few points

    nn = NearestNeighbors(n_neighbors=nb_neighbors + 1) # +1 for self
    nn.fit(points)
    distances, _ = nn.kneighbors(points) 

    # Mean distance to k neighbors (excluding self, distances[:,0] is self-distance if k=0 is included)
    mean_distances = np.mean(distances[:, 1:], axis=1) 
    
    overall_mean = np.mean(mean_distances)
    overall_std = np.std(mean_distances)
    distance_threshold = overall_mean + std_ratio * overall_std
    
    inlier_indices = np.where(mean_distances < distance_threshold)[0]
    
    return _select_by_index(pcl_dict, inlier_indices), inlier_indices


def _segment_plane_ransac(pcl_dict, distance_threshold, n_points_sampled, max_iterations):
    """Segments a plane using RANSAC."""
    if _pcl_is_empty(pcl_dict) or _pcl_len(pcl_dict) < n_points_sampled:
        return np.array([0.0,0.0,0.0,0.0]), np.array([], dtype=int)

    points = pcl_dict['points']
    num_total_points = points.shape[0]
    
    best_inliers_indices = np.array([], dtype=int)
    best_plane_eq = np.array([0.0,0.0,0.0,0.0]) 

    for _ in range(max_iterations):
        sample_indices = random.sample(range(num_total_points), n_points_sampled)
        sampled_points = points[sample_indices]

        p0, p1, p2 = sampled_points[0], sampled_points[1], sampled_points[2]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-9: 
            continue 
        normal = normal / norm_val
        
        a, b, c = normal
        d_coeff = -np.dot(normal, p0) # d from ax+by+cz+d=0
        current_plane_eq = np.array([a, b, c, d_coeff])

        distances = np.abs(np.dot(points, normal) + d_coeff) # Since normal is normalized
        current_inliers_indices = np.where(distances < distance_threshold)[0]

        if len(current_inliers_indices) > len(best_inliers_indices):
            best_inliers_indices = current_inliers_indices
            best_plane_eq = current_plane_eq
            
    return best_plane_eq, best_inliers_indices


def _cluster_dbscan_sklearn(pcl_dict, eps, min_points):
    """Performs DBSCAN clustering using scikit-learn."""
    if _pcl_is_empty(pcl_dict) or _pcl_len(pcl_dict) < min_points: # DBSCAN min_samples includes the point itself
        return np.array([], dtype=int) 

    points = pcl_dict['points']
    # Ensure there are points to cluster
    if points.shape[0] == 0:
        return np.array([], dtype=int)
        
    dbscan = DBSCAN(eps=eps, min_samples=min_points)
    labels = dbscan.fit_predict(points)
    return labels


class BoxEstimator:
    def __init__(self, max_distance):
        self.raw_pcl = _create_empty_pcl()
        self.roi_pcl = _create_empty_pcl()
        self.plane_pcl = _create_empty_pcl()
        self.box_pcl = _create_empty_pcl()
        self.top_side_pcl = _create_empty_pcl()

        self.is_calibrated = False
        self.ground_plane_eq = np.array([0.0,0.0,0.0,0.0]) 

        self.height = None
        self.width = None
        self.length = None

        self.bounding_box = None 
        self.rotation_matrix = np.identity(3) 
        self.translate_vector = np.array([0.0, 0.0, 0.0]) 
        
        self.max_distance = max_distance

    def _reset_dimensions(self):
        self.height = None
        self.width = None
        self.length = None
        self.bounding_box = None
        # self.rotation_matrix = np.identity(3) # Or let it persist from last valid?
        # self.translate_vector = np.array([0.0, 0.0, 0.0]) # Or let it persist?
                                                        # Current logic recomputes these in get_box_top

    def add_visualization_2d(
        self,
        intrinsic_mat, 
        dist_coeffs,   
        annotation_helper: AnnotationHelper,
        image_width: int,
        image_height: int,
    ):
        if self.bounding_box is None or self.height is None or self.height <=0 or \
           self.rotation_matrix is None:
            return

        cam_matrix_np = np.array(intrinsic_mat, dtype=np.float32)
        dist_coeffs_np = np.array(dist_coeffs, dtype=np.float32)
        
        if cam_matrix_np.shape != (3,3):
            print(f"Warning: Intrinsic matrix shape is {cam_matrix_np.shape}, expected (3,3). Skipping visualization.")
            return

        if dist_coeffs_np.ndim == 0: # Scalar
             dist_coeffs_np = np.array([dist_coeffs_np], dtype=np.float32) # Make it 1D
        elif dist_coeffs_np.ndim > 2 : # More than 2D
             print(f"Warning: Distortion coefficients have unsupported shape {dist_coeffs_np.shape}. Skipping visualization.")
             return

        points_floor_transformed = np.hstack((self.bounding_box, np.zeros((4, 1), dtype=np.float64)))
        points_top_transformed = np.hstack((self.bounding_box, self.height * np.ones((4, 1), dtype=np.float64)))
        
        box_points_transformed = np.vstack((points_top_transformed, points_floor_transformed)) 

        inv_rot_mat = np.linalg.inv(self.rotation_matrix)
        
        box_points_world = (inv_rot_mat @ (box_points_transformed - self.translate_vector).T).T
        
        lines = [ 
            [0, 1], [1, 2], [2, 3], [3, 0],  
            [4, 5], [5, 6], [6, 7], [7, 4],  
            [0, 4], [1, 5], [2, 6], [3, 7]   
        ]
        
        cord_change_mat = np.array(
             [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )
        box_points_cv_coord = np.dot(box_points_world, cord_change_mat.T).astype(np.float32) # Ensure float32 for objectPoints

        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)

        if box_points_cv_coord.size == 0: # No points to project
            return

        img_points, _ = cv2.projectPoints(
            box_points_cv_coord, 
            rvec, 
            tvec, 
            cam_matrix_np,    # USE THE CONVERTED NUMPY ARRAY
            dist_coeffs_np    # USE THE CONVERTED NUMPY ARRAY
        )

        for line_indices in lines:
            p1_img = img_points[line_indices[0]].ravel()
            p2_img = img_points[line_indices[1]].ravel()
            
            p1_norm = (float(p1_img[0]) / image_width, float(p1_img[1]) / image_height)
            p2_norm = (float(p2_img[0]) / image_width, float(p2_img[1]) / image_height)
            annotation_helper.draw_line(p1_norm, p2_norm, (255, 0, 0, 255), 2)

    def process_pcl(self, raw_pcl_dai: dai.PointCloudData):
        self._reset_dimensions() 

        pts_xyz_mm = raw_pcl_dai.getPoints() 
        if pts_xyz_mm is None or pts_xyz_mm.size == 0:
            return 0, 0, 0 
        
        pts = pts_xyz_mm.astype(np.float64) * 0.001 

        cols_rgb_u8 = None
        if raw_pcl_dai.isColor():
            try:
                _, cols_rgb_u8_temp = raw_pcl_dai.getPointsRGB() 
                if cols_rgb_u8_temp.shape[1] == 3 and cols_rgb_u8_temp.dtype == np.uint8: # Basic check for color array
                    cols_rgb_u8 = cols_rgb_u8_temp
            except Exception as e:
                print(f"Note: Could not get colors via getPointsRGB directly, attempting separate color fetch or ignoring colors. Error: {e}")
                cols_rgb_u8 = None 

        rgb_colors = np.empty((pts.shape[0] ,3), dtype=np.float64) # Default empty colors matching points count
        if cols_rgb_u8 is not None and cols_rgb_u8.size > 0 and cols_rgb_u8.shape[0] == pts.shape[0]:
            rgb_colors = (cols_rgb_u8[:, :3].astype(np.float64)) / 255.0
        
        current_pcl = {'points': pts, 'colors': rgb_colors}
        
        Rcw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        current_pcl = _rotate_pcl(current_pcl, Rcw)

        current_pcl = _voxel_down_sample(current_pcl, voxel_size=0.01)
        current_pcl, _ = _remove_statistical_outlier(current_pcl, nb_neighbors=30, std_ratio=0.1)
        
        self.raw_pcl = current_pcl

        if _pcl_len(self.raw_pcl) < 100: 
            return 0, 0, 0

        self.crop_plc() 
        if _pcl_len(self.roi_pcl) < 100:
            return 0, 0, 0

        plane_eq_active = np.array([0.0,0.0,0.0,0.0]) # Initialize
        plane_inliers_indices = np.array([], dtype=int)

        if not self.is_calibrated:
            plane_eq_active, plane_inliers_indices = self.detect_ground_plane() 
            if plane_inliers_indices.size == 0 : 
                 return 0,0,0
        else:
            plane_eq_active = self.ground_plane_eq
            if not _pcl_is_empty(self.roi_pcl):
                plane_inliers_indices = self.get_plane_inliers(
                    plane_eq_active, self.roi_pcl['points'], DISTANCE_THRESHOLD_PLANE
                )
            if plane_inliers_indices.size > 0: # Only update self.plane_pcl if inliers found
                self.plane_pcl = _select_by_index(self.roi_pcl, plane_inliers_indices)
            else: # No inliers for calibrated plane, or ROI empty
                self.plane_pcl = _create_empty_pcl()


        box_candidate_pcl = self.get_box_pcl(plane_eq_active, plane_inliers_indices) 
        
        if box_candidate_pcl is None or _pcl_is_empty(box_candidate_pcl):
            return 0, 0, 0
        self.box_pcl = box_candidate_pcl 

        calculated_height = self.get_box_top(plane_eq_active) 
        
        if calculated_height is None or calculated_height <= 0.01: 
            return 0, 0, 0

        # get_dimensions sets self.length, self.width, self.bounding_box using self.top_side_pcl and self.height
        len_dim, wid_dim, hei_dim = self.get_dimensions() 
        
        if len_dim is None or wid_dim is None or hei_dim is None or hei_dim <= 0.01 :
             return 0,0,0 
             
        return len_dim, wid_dim, hei_dim


    def calibrate(self, raw_pcl_dai: dai.PointCloudData):
        self.is_calibrated = False 

        pts_xyz_mm = raw_pcl_dai.getPoints()
        if pts_xyz_mm is None or pts_xyz_mm.size == 0:
            print("Calibration failed: No points in provided PCL data.")
            return False
            
        pts = pts_xyz_mm.astype(np.float64) * 0.001

        cols_rgb_u8 = None
        if raw_pcl_dai.isColor():
            try:
                _, cols_rgb_u8_temp = raw_pcl_dai.getPointsRGB()
                if cols_rgb_u8_temp.shape[1] == 3 and cols_rgb_u8_temp.dtype == np.uint8:
                     cols_rgb_u8 = cols_rgb_u8_temp
            except Exception as e:
                print(f"Note (calibrate): Could not get colors via getPointsRGB. Error: {e}")
                cols_rgb_u8 = None
        
        rgb_colors = np.empty((pts.shape[0],3), dtype=np.float64)
        if cols_rgb_u8 is not None and cols_rgb_u8.size > 0 and cols_rgb_u8.shape[0] == pts.shape[0]:
            rgb_colors = (cols_rgb_u8[:, :3].astype(np.float64)) / 255.0
        
        processed_pcl = {'points': pts, 'colors': rgb_colors}
        
        Rcw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        processed_pcl = _rotate_pcl(processed_pcl, Rcw)
        processed_pcl = _voxel_down_sample(processed_pcl, voxel_size=0.005)
        processed_pcl, _ = _remove_statistical_outlier(processed_pcl, nb_neighbors=30, std_ratio=0.1)

        self.raw_pcl = processed_pcl

        if _pcl_len(self.raw_pcl) < 100:
            print("Calibration failed: Too few points after initial processing.")
            return False

        self.crop_plc() 
        if _pcl_len(self.roi_pcl) < 100:
            print("Calibration failed: Too few points in the ROI.")
            return False
            
        plane_eq, plane_inliers = self.detect_ground_plane() 
        
        if plane_inliers.size == 0 or np.allclose(plane_eq[:3], 0): # Check if normal vector is non-zero
            print("Calibration failed: Could not detect ground plane in ROI.")
            return False

        self.ground_plane_eq = plane_eq
        self.is_calibrated = True
        print(f"Calibration successful. Ground plane equation: {self.ground_plane_eq}")
        return True

    def create_rotation_matrix(self, vec_in_orig, vec_target_orig):
        # Ensure inputs are numpy arrays
        vec_in_orig = np.asarray(vec_in_orig, dtype=np.float64)
        vec_target_orig = np.asarray(vec_target_orig, dtype=np.float64)

        norm_vec_in = np.linalg.norm(vec_in_orig)
        norm_vec_target = np.linalg.norm(vec_target_orig)

        if norm_vec_in < 1e-9 or norm_vec_target < 1e-9: # Handle zero vectors
            return np.identity(3)

        vec_in = vec_in_orig / norm_vec_in
        vec_target = vec_target_orig / norm_vec_target

        v = np.cross(vec_in, vec_target)
        c = np.dot(vec_in, vec_target) 

        if abs(c - 1.0) < 1e-9: 
            R = np.identity(3)
        elif abs(c + 1.0) < 1e-9: 
            # Find an axis k perpendicular to vec_in for 180-degree rotation
            # If vec_in is [0,0,1], k could be [1,0,0]. If vec_in is [1,0,0], k could be [0,1,0].
            # A robust way: find a vector not parallel to vec_in
            tmp_vec = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(vec_in, tmp_vec)) < 1e-6: # vec_in is parallel to tmp_vec
                tmp_vec = np.array([0.0, 1.0, 0.0])
            
            k = np.cross(vec_in, tmp_vec)
            k_norm = np.linalg.norm(k)
            if k_norm < 1e-9: # Should not happen if vec_in is not zero and tmp_vec logic is sound
                 return -np.identity(3) # Fallback for 180 deg flip if axis finding fails (rare)

            k = k / k_norm
            
            # Rodrigues' formula for theta=pi: R = I + 2*K^2, where K is skew-symmetric matrix of k
            # K_skew = [[0, -k_z, k_y], [k_z, 0, -k_x], [-k_y, k_x, 0]]
            # K_skew_sq = k k.T - I (if k is column vector) or outer(k,k) - I (if k is 1D array)
            # R = 2 * np.outer(k,k) - np.identity(3)
            K_skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.identity(3) + 2 * np.dot(K_skew, K_skew) 
        else: 
            v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.identity(3) + v_skew + np.dot(v_skew, v_skew) * (1 / (1 + c))
        
        return R

    def crop_plc(self):
        if _pcl_is_empty(self.raw_pcl):
            self.roi_pcl = _create_empty_pcl()
            return self.roi_pcl # Return the empty PCL
            
        raw_pcl_pts = self.raw_pcl['points']
        if raw_pcl_pts.shape[0] == 0: # Double check
            self.roi_pcl = _create_empty_pcl()
            return self.roi_pcl

        pcl_dist_sq = np.sum(np.square(raw_pcl_pts), axis=1) 
        indices = np.where(pcl_dist_sq < self.max_distance**2)[0]
        
        self.roi_pcl = _select_by_index(self.raw_pcl, indices)
        return self.roi_pcl

    def detect_ground_plane(self): 
        if _pcl_is_empty(self.roi_pcl): # Ensure ROI PCL is not empty
            self.plane_pcl = _create_empty_pcl()
            return np.array([0.0,0.0,0.0,0.0]), np.array([], dtype=int)

        plane_eq, plane_inliers_indices = _segment_plane_ransac(
            self.roi_pcl, 
            DISTANCE_THRESHOLD_PLANE, 
            N_POINTS_SAMPLED_PLANE_RANSAC, 
            MAX_ITER_PLANE_RANSAC
        )
        
        if plane_inliers_indices.size > 0:
            self.plane_pcl = _select_by_index(self.roi_pcl, plane_inliers_indices)
        else:
            self.plane_pcl = _create_empty_pcl()
        return plane_eq, plane_inliers_indices

    def get_box_pcl(self, ground_plane_eq_ignored, ground_plane_inliers_indices): 
        # ground_plane_eq_ignored is not directly used here, inliers are key
        if _pcl_is_empty(self.roi_pcl): # Ensure ROI PCL is not empty
            return None
            
        outliers_from_plane = _select_by_index(self.roi_pcl, ground_plane_inliers_indices, invert=True)

        # Ensure min_points for DBSCAN is reasonable; if outliers_from_plane has fewer points, DBSCAN might behave unexpectedly or error.
        # _cluster_dbscan_sklearn already checks if _pcl_len < min_points.
        if _pcl_is_empty(outliers_from_plane): 
            return None 

        labels = _cluster_dbscan_sklearn(outliers_from_plane, eps=0.02, min_points=10)
        
        # Filter out noise points (label -1)
        valid_labels = labels[labels != -1]
        if valid_labels.size == 0: # No non-noise clusters
            return None
            
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        if counts.size == 0: 
            return None
        
        largest_cluster_label = unique_labels[counts.argmax()]
        # Indices are relative to `outliers_from_plane`
        box_indices_in_outliers = np.where(labels == largest_cluster_label)[0]
        
        if box_indices_in_outliers.size == 0:
            return None
            
        box_pcl_candidate = _select_by_index(outliers_from_plane, box_indices_in_outliers)
        return box_pcl_candidate


    def get_box_top(self, plane_eq_ground): 
        normal_vec_ground = np.asarray(plane_eq_ground[0:3], dtype=np.float64)
        d_ground = float(plane_eq_ground[3])

        if np.linalg.norm(normal_vec_ground) < 1e-6:
            self.height = None
            self.top_side_pcl = _create_empty_pcl()
            return None

        target_up_vector = np.array([0.0, 0.0, 1.0]) 
        self.rotation_matrix = self.create_rotation_matrix(normal_vec_ground, target_up_vector)
        
        # After rotation, the ground plane's normal becomes target_up_vector [0,0,1].
        # The equation of the plane normal_vec_ground . X + d_ground = 0 becomes:
        # (R @ normal_vec_ground) . X_rot + d_ground = 0  (where R is self.rotation_matrix)
        # target_up_vector . X_rot + d_ground = 0  => z_rot + d_ground = 0
        # So, points on the rotated ground plane have z_rot = -d_ground.
        avg_z_of_rotated_ground_plane = -d_ground 

        self.translate_vector = np.array([0.0, 0.0, -avg_z_of_rotated_ground_plane]) # Makes ground z_rot_trans = 0

        # Apply transformations. Create copies if original PCLs should remain untouched before this step.
        # Assuming self.raw_pcl, self.plane_pcl, self.box_pcl are meant to be transformed here.
        if not _pcl_is_empty(self.raw_pcl):
            self.raw_pcl = _translate_pcl(_rotate_pcl(self.raw_pcl, self.rotation_matrix), self.translate_vector)
        if not _pcl_is_empty(self.plane_pcl): # self.plane_pcl contains points on the ground plane
            self.plane_pcl = _translate_pcl(_rotate_pcl(self.plane_pcl, self.rotation_matrix), self.translate_vector)
        if not _pcl_is_empty(self.box_pcl): 
            self.box_pcl = _translate_pcl(_rotate_pcl(self.box_pcl, self.rotation_matrix), self.translate_vector)
        
        if _pcl_is_empty(self.box_pcl): 
             self.height = None; self.top_side_pcl = _create_empty_pcl(); return None

        points_np_box_transformed = self.box_pcl['points']
        
        top_plane_normal_target = np.array([0.0,0.0,1.0]) # Top plane normal in transformed space
        top_plane_eq, top_plane_inliers_idx = self.fit_plane_vec_constraint(
            top_plane_normal_target, points_np_box_transformed, thresh=0.03, n_iterations=100 # Reduced iterations for speed
        )

        if top_plane_inliers_idx.size == 0 or len(top_plane_eq) < 4: # Check if plane fitting was successful
            self.height = None; self.top_side_pcl = _create_empty_pcl(); return None

        self.top_side_pcl = _select_by_index(self.box_pcl, top_plane_inliers_idx)
        # top_plane_eq is [0,0,1,d_top]. Height is abs(d_top) because ground is at z=0.
        # z_top + d_top = 0 => z_top = -d_top. Height = z_top - z_ground = -d_top - 0.
        self.height = abs(top_plane_eq[3]) 
        
        return self.height


    def get_dimensions(self): 
        if _pcl_is_empty(self.top_side_pcl) or self.height is None or self.height <= 0:
            self.length = 0.0; self.width = 0.0; self.bounding_box = None;
            # Return consistent types
            return 0.0, 0.0, (float(self.height) if self.height is not None else 0.0)

        coordinates_2d = self.top_side_pcl['points'][:, 0:2].astype("float32")
        
        if coordinates_2d.shape[0] < 3: 
            self.length = 0.0; self.width = 0.0; self.bounding_box = None;
            return 0.0, 0.0, float(self.height)

        rect = cv2.minAreaRect(coordinates_2d) 
        self.bounding_box = cv2.boxPoints(rect).astype(np.float64) # Store as float64 to match other point data
        
        dim1, dim2 = rect[1] # These are width and height of the 2D rectangle
        self.width = float(min(dim1, dim2)) # Assign smaller to width
        self.length = float(max(dim1, dim2)) # Assign larger to length
        
        return self.length, self.width, float(self.height)


    def fit_plane_vec_constraint(self, norm_vec_target, pts_np, thresh=0.05, n_iterations=300):
        best_eq = np.array([0.0,0.0,0.0,0.0]) # Default empty/invalid plane
        best_inliers_indices = np.array([], dtype=int)
        n_total_points = pts_np.shape[0]

        if n_total_points == 0:
            return best_eq, best_inliers_indices
        
        norm_vec_target_unit = norm_vec_target / np.linalg.norm(norm_vec_target)

        for _ in range(n_iterations):
            # Randomly sample one point from pts_np
            id_sample = random.choice(range(n_total_points))
            point_on_plane = pts_np[id_sample]
            
            # Plane equation: norm_vec . P + d = 0 => d = - (norm_vec . P)
            d_coeff = -np.dot(norm_vec_target_unit, point_on_plane) 
            current_plane_eq = np.concatenate((norm_vec_target_unit, [d_coeff]))
            
            current_inliers_indices = self.get_plane_inliers(current_plane_eq, pts_np, thresh)
                                                                
            if len(current_inliers_indices) > len(best_inliers_indices):
                best_eq = current_plane_eq
                best_inliers_indices = current_inliers_indices
        
        return best_eq, best_inliers_indices

    def get_plane_inliers(self, plane_eq_np, pts_np, thresh=0.05):
        if pts_np.shape[0] == 0 or len(plane_eq_np) < 4:
            return np.array([], dtype=int)
            
        dist_pt = self.get_pts_distances_plane(plane_eq_np, pts_np)
        # dist_pt can be empty if pts_np was empty or plane invalid
        if dist_pt.size == 0:
            return np.array([], dtype=int)
            
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        return pt_id_inliers

    def get_pts_distances_plane(self, plane_eq_np, pts_np):
        if pts_np.shape[0] == 0 or len(plane_eq_np) < 4:
            return np.array([]) # Return empty array if no points or invalid plane equation

        normal_vec = np.asarray(plane_eq_np[0:3], dtype=np.float64)
        d_coeff = float(plane_eq_np[3])
        
        norm_of_normal = np.linalg.norm(normal_vec)
        if norm_of_normal < 1e-9: # Avoid division by zero for invalid normal
            return np.full(pts_np.shape[0], np.inf) 
            
        # Distance = (normal_vec . Pts + d_coeff) / ||normal_vec||
        dist_pt = (np.dot(pts_np, normal_vec) + d_coeff) / norm_of_normal
        return dist_pt
