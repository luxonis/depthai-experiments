import numpy as np
import open3d as o3d
from depthai_sdk.logger import LOGGER
from typing import Tuple
import time

# thr = 40, voxel_size = 20, iter = 300
class GroundPlaneEstimator:
    def __init__(self, distance_threshold=40, n_points_sampled_plane=3, max_iterations=300, voxel_size=25):
        
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
        self.n_points_sampled_plane = n_points_sampled_plane
        self.voxel_size_downsample = voxel_size

        self.ground_plane_eq = None 
        self.inliner_thr = 0.5

        self.objects_points = None
        self.plane_points = None

        self.pcd = o3d.geometry.PointCloud()
        self.points_buffer = np.empty((0, 3), dtype=np.float64)     # Buffer for point cloud data

        # For drawing plane mesh
        self.new = True 
        self.positions = None
        self.indices = None
        self.normals = None

    def estimate_ground_plane(self, points, gravity_vect):

        """
        Estimate the ground plane from a point cloud using RANSAC

        Args:
            points: Point cloud data (Nx3 numpy array).
            gravity_vect: Gravity vector data from the IMU (1D numpy array) 

        Returns:
            plane_eq: Ground plane equation (A, B, C, D) or None if estimation fails.
            plane_points: Ground plane points (nx3 numpy array)
            objects_points: Remaining points, outliers (nx3 numpy array)
            success: Boolean indicating if the estimation was successful.
        """

        # Update point cloud data
        self.update_point_cloud(points)

        # Downsample the point cloud
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self.voxel_size_downsample)

        # Remove outliers
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)
       
        current_points = np.asarray(self.pcd.points)
        current_points_count = len(current_points)

        if self.ground_plane_eq is not None:
            # Check if we need new plane model based on inlier ratio
            outliers, inliers = self.get_outliers(current_points)
            current_inliers_count = len(inliers)
            current_inlier_ratio = current_inliers_count / current_points_count
            #print('New inliers ratio: ', current_inlier_ratio)
            if current_inlier_ratio > self.inliner_thr:
                # No significant change in inlier ratio, keep the existing plane
                #print('Using previous ground plane equation.')
                self.plane_points = inliers
                self.objects_points = outliers
                self.new = False
                success = True
                return self.ground_plane_eq, self.plane_points, self.objects_points, success
        
        # If no previous ground plane or inlier ratio has changed significantly, calculate a new model
        #print('Estimating new ground plane...')
        
        # Perform RANSAC plane segmentation
        plane_eq, plane_inliers = self.pcd.segment_plane(self.distance_threshold, self.n_points_sampled_plane, self.max_iterations)
        
        # Calculate inlier percentage
        inlier_count = len(plane_inliers)
        inlier_ratio = inlier_count / len(self.pcd.points)
        #print('Inliers ratio: ', inlier_ratio)

        if inlier_ratio >= 0.6:
            # Check if estimated plane is the ground 
            if self.check_if_ground(gravity_vect, plane_eq):

                self.ground_plane_eq = plane_eq
                # Store the inliers as ground points and the outliers as objects_points
                mask = np.zeros(len(current_points), dtype=bool)
                mask[plane_inliers] = True

                self.plane_points = current_points[mask]
                self.objects_points = current_points[~mask]
                self.new = True     # for drawing plane mesh
                success = True
            else:
                # Estimated plane is not the ground 
                self.ground_plane_eq = None
                self.plane_points = None
                self.objects_points = None
                success = False
        else:
            #print("Not enough inliers for ground estimation!")
            self.ground_plane_eq = None
            self.plane_points = None
            self.objects_points = None
            success = False

        return self.ground_plane_eq, self.plane_points, self.objects_points, success
    
    def get_plane_mesh(self, size=10, divisions=10) -> Tuple:
        """
        Create a mesh representation of a plane given its equation Ax + By + Cz + D = 0.

        Args:
            size: Size of the plane 
            divisions: Number of divisions in each dimension 

        Returns:
            positions: List of 3D points
            indices: List of indices for triangles
            normals: List of normal vectors for each vertex
        """
        # Calculate only if new ground plane, else use old values 
        if self.new == True:
            # Normalize the normal vector
            A,B,C,D = self.ground_plane_eq
            normal = np.array([A, B, C])
            normal = normal / np.linalg.norm(normal)

            # Create a grid of points
            x = np.linspace(-size/2, size/2, divisions)
            y = np.linspace(-size/2, size/2, divisions)
            X, Y = np.meshgrid(x, y)

            # Calculate Z values
            if C != 0:
                Z = (-A*X - B*Y - D) / C
            elif B != 0:
                Z = (-A*X - C*Y - D) / B
                Y, Z = Z, Y
            else:
                Z = (-B*Y - C*Z - D) / A
                X, Z = Z, X

            # Create positions
            positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3).tolist()

            # Create indices
            indices = []
            for i in range(divisions - 1):
                for j in range(divisions - 1):
                    square_start = i * divisions + j
                    indices.extend([
                        square_start, square_start + 1, square_start + divisions,
                        square_start + 1, square_start + divisions + 1, square_start + divisions
                    ])

            # Create normals
            normals = [normal.tolist() for _ in range(len(positions))]

            self.positions = positions
            self.indices = indices
            self.normals = normals

        return self.positions, self.indices, self.normals
    
    def get_outliers(self, points) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get outliers and inliers from a point cloud
        """
        A,B,C,D = self.ground_plane_eq
        # Calculate the denominator (constant for all points)
        denominator = np.sqrt(A**2 + B**2 + C**2)
        threshold = 30 # mm
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / denominator
        return (points[distances > threshold], points[distances <= threshold])
    
    def get_ground_plane(self):
        return self.ground_plane_eq

    def update_point_cloud(self, points):
        """ 
        Updates the Open3D point cloud data using an internal buffer, which makes execution faster. 
        """

        points = points.reshape(-1, 3)  

        # Check if we need to resize the buffer
        if points.shape[0] > self.points_buffer.shape[0]:
            self.points_buffer = np.empty((points.shape[0], 3), dtype=np.float64)

        np.copyto(self.points_buffer[:points.shape[0]], points)
        self.pcd.points = o3d.utility.Vector3dVector(self.points_buffer[:points.shape[0]])
    
    def check_if_ground(self, gravity_vect, plane_eq):
        """ 
        Checks if the plane represents the ground using the gravity vector from the IMU.
        If the plane is the ground, the normal of the plane and the gravity vector should be parallel.
        """

        plane_normal = plane_eq[:3]
        plane_normal_norm = plane_normal / np.linalg.norm(plane_normal)

        # Camera and IMU c.s are not aligned; Align gravity vector with camera c.s
        theta = np.deg2rad(180)
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,             0,            1]])
        aligned_gravity_vect = np.dot(R_z, gravity_vect)
        aligned_gravity_vect_norm = aligned_gravity_vect / np.linalg.norm(aligned_gravity_vect)

        # If the vectors are parallerl dot product should be -1
        dot_product = np.dot(plane_normal_norm, aligned_gravity_vect_norm)
        if np.isclose(dot_product, -1, atol=0.15):  # Allow some tolerance
            print("The estimated plane is likely the ground.")
            return True
        else:
            print("The estimated plane not the ground.")
            return False