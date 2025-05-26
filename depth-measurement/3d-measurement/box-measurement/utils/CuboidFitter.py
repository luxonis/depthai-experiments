import numpy as np
import open3d as o3d
from itertools import combinations
import time

class CuboidFitter:
    def __init__(self, distance_threshold=12, sample_points=3, max_iterations=500, voxel_size=10):
        self.distance_threshold = distance_threshold
        self.sample_points = sample_points
        self.max_iterations = max_iterations
        self.voxel_size = voxel_size
        self.orthogonality_thr = 0.2
        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.point_cloud = o3d.geometry.PointCloud()
        self.points_buffer = np.empty((0, 3), dtype=np.float64)     # Buffer for point cloud data
        self.line_set = o3d.geometry.LineSet()
        self.reset()

    def update_point_cloud(self, points):
        """ 
        Updates the Open3D point cloud data using an internal buffer, which makes execution faster. 
        """

        # commented out for xlinkin script
        #points = points.reshape(-1, 3)  

        # Check if we need to resize the buffer
        if points.shape[0] > self.points_buffer.shape[0]:
            self.points_buffer = np.empty((points.shape[0], 3), dtype=np.float64)

        np.copyto(self.points_buffer[:points.shape[0]], points)
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points_buffer[:points.shape[0]])
        #print(points.shape)
        #print(np.asarray(self.point_cloud.points).shape)

    def reset(self):
        """
        Reset the fitter for a new frame.
        """
        #self.point_cloud = None
        self.all_planes = None
        self.corners = None
        self.center = None
        self.planes = []
        self.plane_meshes = []
        self.plane_points = []
    
    def set_point_cloud(self, pcl_points, timing_results, colors=None):
        if timing_results:
            #start_time2 = time.time()
            self.center = pcl_points.mean(axis=0)
            self.update_point_cloud(pcl_points)
            #print('UPDATE: ', time.time() - start_time2)

            #start_time2 = time.time()
            if colors is not None:
                # Normalize colors to [0, 1] range and set them
                self.colors = colors / 255.0  # Assuming input is in [0, 255]
                self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
            #print('COLOR: ', time.time() - start_time2)

            # filtering 
            start_time = time.time()
            self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            #print('voxel downsample: ', time.time() - start_time)
            timing_results["pcl_voxel_downsample"].append(time.time() - start_time)
            start_time = time.time()
            
            self.point_cloud = self.point_cloud.remove_statistical_outlier(40, 0.1)[0] 
            #print('remove outlier: ', time.time() - start_time)
            timing_results["pcl_remove_statistical_outlier"].append(time.time() - start_time)
        
            start_time = time.time()
            filtered_points, filtered_colors = self.MAD_filtering(self.point_cloud)
            #print('FILTER MAD: ', time.time() - start_time2)
            #start_time2 = time.time()
            self.point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
            self.point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            timing_results["pcl_MAD_filtering"].append(time.time() - start_time)
        else:
            self.center = pcl_points.mean(axis=0)
            self.update_point_cloud(pcl_points)
            if colors is not None:
            # Normalize colors to [0, 1] range and set them
                self.colors = colors / 255.0  # Assuming input is in [0, 255]
                self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
            self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            self.point_cloud = self.point_cloud.remove_statistical_outlier(40, 0.1)[0] 
            filtered_points, filtered_colors = self.MAD_filtering(self.point_cloud)
            self.point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
            self.point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        #print('FILTER MAD SET: ', time.time() - start_time2)

    def MAD_filtering(self, pcl, k=3):

        colors = np.asarray(pcl.colors)
        points = np.asarray(pcl.points)    # Compute the median point (robust center estimate)
        median_point = np.median(points, axis=0)    # Compute the Euclidean distance of each point from the median point
        
        distances = np.linalg.norm(points - median_point, axis=1)    # Compute the median of these distances
        median_distance = np.median(distances)    # Compute the Median Absolute Deviation (MAD)
        
        mad = np.median(np.abs(distances - median_distance))    # Create a mask for points within the threshold * MAD
        mask = np.abs(distances - median_distance) < k * mad    # Filter the points using the mask
        
        #robust_sigma = 1.4826 * mad  # scale factor for normality

        # 6) Outlier mask
        #mask = (distances < (median_distance + k*robust_sigma))
        filtered_points = points[mask]    # Create a new point cloud for the filtered points
        filtered_colors = colors[mask] 
        #filtered_pcd = o3d.geometry.PointCloud()
        #filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)         

        return filtered_points, filtered_colors

    def distance_between_planes(self, plane1, plane2):
        normal1 = np.array(plane1[:3])
        normal2 = np.array(plane2[:3])
        norm1 = np.linalg.norm(normal1)
        norm2 = np.linalg.norm(normal2)
        if not np.allclose(normal1 / norm1, normal2 / norm2, atol=1e-6):
            print("Planes are not parallel")
            return None
        return abs(plane2[3] - plane1[3]) / norm1
    
    def translate_plane(self, plane_eq, distance):
        normal = plane_eq[:3]
        d = plane_eq[3]
        s = -1 if normal[2] < 0 else 1
        translated_d = d - s * distance * np.linalg.norm(normal)
        return [normal[0], normal[1], normal[2], translated_d]
    
    def translate_planes(self, distances):
        # for multiple planes at once
        planes = np.asarray(self.planes)
        normal = planes[:, :3]  # Extract the normal vectors
        norm = np.linalg.norm(normal, axis=1)  # Compute the norm of each normal
        s = np.sign(normal[:, 2])  # Get the direction for translation based on z-component
        translated_d = planes[:, 3] - s * distances * norm  # Calculate the translated d values
        translated_planes = np.column_stack((normal, translated_d))  # Combine the normal and new d values
        return translated_planes
     
    def intersect_planes(self, plane_eq1, plane_eq2, plane_eq3, tol=1e-6):
        A = np.array([plane_eq1[:3], plane_eq2[:3], plane_eq3[:3]])
        B = -np.array([plane_eq1[3], plane_eq2[3], plane_eq3[3]])
        # Calculate determinant and check if it's close to zero (nearly parallel planes)
        det_A = np.linalg.det(A)
        if np.abs(det_A) < tol:
            print("intersect_planes: Planes are nearly parallel or coincident. Cannot find intersection.")
            return None

        # Solve for the intersection point
        intersection_point = np.linalg.solve(A, B)
        
        return intersection_point

    def dist_to_plane1(self, point, plane_eq):
        return abs(np.dot(plane_eq[:3], point) + plane_eq[3]) / np.linalg.norm(plane_eq[:3])
    
    def dist_to_plane(self, points, plane_eq):
        # for multiple points 
        normal = np.array(plane_eq[:3])  
        dot_product = np.dot(points, normal) 
        distances = np.abs(dot_product + plane_eq[3]) / np.linalg.norm(normal)
        
        return distances

    def fit_plane(self):
        if len(self.point_cloud.points) < self.sample_points:
            return None, None, False
        plane_eq, plane_inliers = self.point_cloud.segment_plane(
            self.distance_threshold, self.sample_points, self.max_iterations)
        inlier_count = len(plane_inliers)
        inlier_ratio = inlier_count / len(self.point_cloud.points)
        #print(inlier_ratio)
        if inlier_ratio >= 0.2:
            return plane_eq, plane_inliers, True
        print('Not enough inliers for plane fitting!')
        return None, None, False
    
    def check_orthogonal(self, plane_eq1, plane_eq2):
        normal1 = np.array(plane_eq1[:3]) / np.linalg.norm(plane_eq1[:3])
        normal2 = np.array(plane_eq2[:3]) / np.linalg.norm(plane_eq2[:3])
        #print('Dot product of fitted planes: ', np.dot(normal1, normal2))
        return np.isclose(np.dot(normal1, normal2), 0, atol=self.orthogonality_thr)
    
    def get_plane_mesh(self, plane_eq, size=500, divisions=10):
        A, B, C, D = plane_eq
        normal = np.array([A, B, C]) / np.linalg.norm([A, B, C])
        x = np.linspace(-size / 2, size / 2, divisions)
        y = np.linspace(-size / 2, size / 2, divisions)
        X, Y = np.meshgrid(x, y)

        if C != 0:
            Z = (-A * X - B * Y - D) / C
        else:
            return None

        positions = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        indices = []
        for i in range(divisions - 1):
            for j in range(divisions - 1):
                start = i * divisions + j
                indices.extend([
                    [start, start + 1, start + divisions],
                    [start + 1, start + divisions + 1, start + divisions],
                ])
        normals = np.repeat([normal], len(positions), axis=0)
        
        return positions, indices, normals 
    
    def fit_orthogonal_planes(self, max_attempts=20):
        if self.point_cloud is None:
            print("No point cloud loaded!")
            return False

        attempts = 0
        while len(self.planes) < 3 and attempts < max_attempts:
            plane_eq, inliers, success = self.fit_plane()
            if not success:
                print("Plane fitting failed. Try adjusting parameters.")
                return False
            #self.point_cloud.paint_uniform_color([1, 0, 0])
            #o3d.visualization.draw_geometries([self.point_cloud] + self.plane_points)

            attempts += 1  # Increment attempt after each plane fitting attempt

            if all(self.check_orthogonal(plane_eq, existing) for existing in self.planes):
                self.planes.append(plane_eq)
                points = self.point_cloud.select_by_index(inliers)
                #points.paint_uniform_color(self.colors[len(self.planes) - 2])
                self.plane_points.append(points)
                self.point_cloud = self.point_cloud.select_by_index(inliers, invert=True)  # Remove inliers
                #o3d.visualization.draw_geometries(self.plane_points)
            #else:
                #print("Plane is not orthogonal to previously fitted planes. Retrying...")

        if len(self.planes) < 3:
            print("Max attempts reached. Could not fit all orthogonal planes.")
            return False

        return True
    
    def calculate_dimensions_corners_MAD(self):
        # With MAD filtering 
        # downsample
        for i, pcl in enumerate(self.plane_points):
            self.plane_points[i] = pcl.voxel_down_sample(voxel_size=self.voxel_size)

        distances = np.zeros(len(self.planes))
        #distances2 = np.zeros(len(self.planes))
        for i, plane in enumerate(self.planes):
            combined_points = np.vstack(
                [np.asarray(pts.points) for j, pts in enumerate(self.plane_points) if j != i]
            )
            plane_distances = self.dist_to_plane(combined_points, plane)
            median_dist = np.median(plane_distances)
            mad = np.median(np.abs(plane_distances - median_dist))

            # Define a threshold (e.g., 3 MADs from the median)
            threshold = median_dist + 2 * mad

            # Select the point closest to this threshold
            filtered_distances = [dist for dist in plane_distances if dist <= threshold]
            distances[i] = np.max(filtered_distances)
        
        translated_planes = self.translate_planes(distances)

        self.all_planes = np.vstack((self.planes, translated_planes))

        self.corners = []
        for plane_comb in combinations(self.all_planes, 3):
            point = self.intersect_planes(*plane_comb)
            if point is not None:
                self.corners.append(point)
        print("Number of corners found: ", len(self.corners))

        dimensions = [
            self.distance_between_planes(self.planes[i], translated_planes[i]) / 10.0 for i in range(3)
        ]
        print("Dimensions: ", dimensions)

        # Assuming length > width > height (can be cases where this is not true..) TO DO: other way of sorting
        sorted_dims = np.sort(dimensions)[::-1]

        return sorted_dims, np.array(self.corners)
    
    def calculate_dimensions_corners(self):
        # downsample
        for i, pcl in enumerate(self.plane_points):
            self.plane_points[i] = pcl.voxel_down_sample(voxel_size=self.voxel_size)

        distances = np.zeros(len(self.planes))
        for i, plane in enumerate(self.planes):
            combined_points = np.vstack(
                [np.asarray(pts.points) for j, pts in enumerate(self.plane_points) if j != i]
            )
            plane_distances = self.dist_to_plane(combined_points, plane)
            distances[i] = np.max(plane_distances)

        
        translated_planes = self.translate_planes(distances)

        self.all_planes = np.vstack((self.planes, translated_planes))

        self.corners = []
        for plane_comb in combinations(self.all_planes, 3):
            point = self.intersect_planes(*plane_comb)
            if point is not None:
                self.corners.append(point)

        dimensions = [
            self.distance_between_planes(self.planes[i], translated_planes[i]) / 10.0 for i in range(3)
        ]

        # Assuming length > width > height (can be cases where this is not true..) TO DO: other way of sorting
        sorted_dims = np.sort(dimensions)[::-1]

        return sorted_dims, np.array(self.corners)
    
    def refine_box(self, corners):
        """
        Refines the 8 corners of a 3D box to ensure it forms a perfect rectangular box.
        """
        # Validate input
        if len(corners) != 8:
            raise ValueError("Expected 8 corners, but got {}.".format(len(corners)))

        # Convert corners to a NumPy array for easier manipulation
        corners = np.array(corners)

        # Calculate the centroid of the box
        centroid = np.mean(corners, axis=0)

        # Subtract centroid to work in local coordinates
        local_corners = corners - centroid

        # Perform PCA to find the principal axes
        u, s, vh = np.linalg.svd(local_corners)
        principal_axes = vh[:3]  # 3 principal axes (orthogonal basis)

        # Project the corners onto the principal axes
        projected_corners = np.dot(local_corners, principal_axes.T)

        # Determine the bounding box dimensions along each axis

        min_vals = np.min(projected_corners, axis=0)
        max_vals = np.max(projected_corners, axis=0)

        # Reconstruct the 8 corners of the perfect box in local coordinates
        refined_corners = []
        for x in [min_vals[0], max_vals[0]]:
            for y in [min_vals[1], max_vals[1]]:
                for z in [min_vals[2], max_vals[2]]:
                    refined_corners.append([x, y, z])

        # Transform back to the original coordinate space
        refined_corners = np.dot(refined_corners, principal_axes) + centroid

        return np.array(refined_corners)
    
    def get_3d_lines(self, corners):
        # Validate input
        if len(corners) != 8:
            raise ValueError("Expected 8 corners, but got {}.".format(len(corners)))

        # Sort corners in order for a rectangluar box 
        sorted_corners = self.sort_corners(corners)
        #refined_corners = self.refine_box(sorted_corners)

        #print(sorted_corners)

        # Define correct connections
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom plane
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top plane
            [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical connections
        ]

        # Create line segments
        line_segments = []
        for line in lines:
            line_segments.append(sorted_corners[line[0]])
            line_segments.append(sorted_corners[line[1]])

        return np.array(line_segments)
    
    def get_3d_lines_o3d(self, corners):
        """
        Generate 3D line segments for the edges of a rectangular cuboid given its corners.
        """

        # Create Open3D LineSet
        
        sorted_corners = self.sort_corners(corners)
        # Assign points and lines
        self.line_set.points = o3d.utility.Vector3dVector(sorted_corners)
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom plane
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top plane
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical connections
        ]
        self.line_set.lines = o3d.utility.Vector2iVector(connections) 
        self.line_set.paint_uniform_color([1, 0, 0])

        return self.line_set
        
    def create_R_from_normals(self):
        """
        Creates rotation matrix from normal vectors of 3 ortogonal planes (WIP)
        """

        normals = np.array(self.planes)[:, :3]
        normals = normals / np.linalg.norm(normals, axis=1)

        # Global axes for reference
        global_z = np.array([0, 0, 1])
        global_x = np.array([1, 0, 0])

        # Step 1: Find the normal closest to the Z-axis
        z_idx = np.argmax([np.abs(np.dot(normal, global_z)) for normal in normals])
        z_normal = normals[z_idx]
        
        # Step 2: Find the normal closest to the X-axis (excluding the Z-axis)
        remaining_indices = [i for i in range(3) if i != z_idx]
        x_idx = max(remaining_indices, key=lambda i: np.abs(np.dot(normals[i], global_x)))
        x_normal = normals[x_idx]

        # Step 3: Assign the remaining normal to Y-axis
        y_idx = [i for i in remaining_indices if i != x_idx][0]
        y_normal = normals[y_idx]

        # Step 4: Construct the rotation matrix
        rotation_matrix = np.column_stack((x_normal, y_normal, z_normal))

        return rotation_matrix
    
    def sort_plane_clockwise(self, plane):
        """
        Sorts a plane's points in a consistent order (e.g., clockwise) based on the centroid.

        Args:
            plane (np.ndarray): 4x3 array of points in the plane.

        Returns:
            np.ndarray: Sorted 4x3 array of points in clockwise order.
        """
        center = np.mean(plane[:, :2], axis=0)  # Compute centroid in XY plane
        angles = np.arctan2(plane[:, 1] - center[1], plane[:, 0] - center[0])  # Angles to centroid
        return plane[np.argsort(angles)]  # Sort points by angle
    
    def sort_corners(self, corners):
        R = self.create_R_from_normals()
        # R is not quite orthogonal, since there is tol for planes orthogonality 0.2 
        U, _, Vt = np.linalg.svd(R)
        R_orthogonal = np.dot(U, Vt)
        
        # Align corners with global c.s for better sorting 
        centered_corners = corners - self.center
        rotated_corners = np.dot(centered_corners, R_orthogonal)

        # Sort in top and bottom plane by z value 
        sorted_indices = np.argsort(rotated_corners[:, 2])  # Sort by Z
        bottom_plane = rotated_corners[sorted_indices[:4]]  # First 4 points -> bottom
        top_plane = rotated_corners[sorted_indices[4:]]

        # Sort the corners in clockwise order 
        bottom_plane = self.sort_plane_clockwise(bottom_plane)
        top_plane = self.sort_plane_clockwise(top_plane)
        sorted_corners = np.vstack([bottom_plane, top_plane])

        # Transform back to original coordinates 
        back_rotated = np.dot(sorted_corners, R_orthogonal.T)
        back = back_rotated + self.center

        return back






    
