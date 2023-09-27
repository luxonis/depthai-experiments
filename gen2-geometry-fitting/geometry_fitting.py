import os
import time
import json
from sys import maxsize
from itertools import combinations
from typing import *

from numba import jit
import numpy as np
import open3d as o3d
import cv2
import depthai as dai
from pathlib import Path
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
import blobconverter


class PointCloudVisualizer():
    """
    PointCloudVisualizer is a utility class designed to facilitate the visualization of point clouds using Open3D.
    It provides functionalities to convert RGBD images to point clouds, visualize them in a window, and manipulate
    the visualized point clouds by adding meshes or clearing them. The class also handles camera intrinsics 
    and sets up the visualization environment with appropriate parameters for optimal viewing.
    
    Attributes:
    - R_camera_to_world: Rotation matrix to transform camera coordinates to world coordinates.
    - pcl: The main point cloud object that will be visualized.
    - pinhole_camera_intrinsic: Camera intrinsic parameters for projecting depth to 3D points.
    - vis: The Open3D visualizer object.
    - meshes: A list to store additional mesh geometries added to the visualizer.
    """    
    
    
    
    def __init__(self, intrinsic_matrix: np.ndarray, width: int, height: int) -> None:
        # Rotation matrix to transform from camera to world coordinates
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        
        # Create an empty point cloud object
        self.pcl = o3d.geometry.PointCloud()
        
        # Define the camera intrinsic parameters using the provided matrix
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        
        # Initialize the Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")
        
        # Add the point cloud to the visualizer
        self.vis.add_geometry(self.pcl)
        
        # Add a coordinate frame origin to the visualizer for reference
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        self.vis.add_geometry(origin)
        
        # Adjust the view control settings
        view_control = self.vis.get_view_control()
        view_control.set_constant_z_far(1000)

        # List to store all the meshes
        self.meshes = []
        
        # Adjust render options to show the back face of the mesh
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True

        
        
        
    def rgbd_to_projection(self, depth_map: np.ndarray, rgb: np.ndarray, downsample: bool = False, remove_noise: bool = False) -> o3d.geometry.PointCloud:
        """
        Convert RGBD images to a point cloud projection.

        Parameters:
        - depth_map: Depth image.
        - rgb: RGB image.
        - downsample (optional): Boolean flag to determine if the point cloud should be downsampled.
          False is recommended as curved geometry tends to be detected poorly otherwise
        - remove_noise (optional): Boolean flag to determine if noise should be removed from the point cloud. 
          False is recommended as this is very slow

        Returns:
        - Point cloud generated from the RGBD image.
        """

        # Convert numpy RGB and depth images to Open3D Image format
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        # Create an RGBD image from the RGB and depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, 
            convert_rgb_to_intensity=(len(rgb.shape) != 3),  # Convert RGB to intensity if it's not already in that format
            depth_trunc=20000,  # Truncate depth values beyond 20 meters
            depth_scale=1000.0  # Scale factor for depth values
        )

        # Create a point cloud from the RGBD image using the camera intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        # Downsample the point cloud if the flag is set
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Remove noise from the point cloud if the flag is set
        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

        # Update the point cloud object of the class with the new points and colors
        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors

        # Rotate the point cloud to align with the world coordinates
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))

        return self.pcl
    
        

    def visualize_pcd(self) -> None:
        """
        Visualize the point cloud in the visualizer window.
        """

        # Get the view control object from the visualizer
        view_control = self.vis.get_view_control()

        # Convert the current view control parameters to pinhole camera parameters
        pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()

        # Set the extrinsic matrix for the camera parameters.
        # This matrix is hardcoded due to observed unexpected behavior with other values.
        pinhole_camera_parameters.extrinsic = [[1.0, 0.0, 0.0, -0.141],
                                               [0.0, -1.0, 0.0, 0.141],
                                               [0.0, 0.0, -1.0, 0.52655451],
                                               [0.0, 0.0, 0.0, 1.0]]

        # Apply the updated camera parameters to the visualizer's view control
        view_control.convert_from_pinhole_camera_parameters(pinhole_camera_parameters)

        # Update the point cloud geometry in the visualizer
        self.vis.update_geometry(self.pcl)

        # Poll for any events (like user input)
        self.vis.poll_events()

        # Update the visualizer's renderer to reflect the changes
        self.vis.update_renderer()
        
        

    def close_window(self) -> None:
        """
        Close the visualizer window.
        """
        self.vis.destroy_window()
        


    def add_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        Add a mesh to the visualizer and store it in the list of meshes.

        Parameters:
        - mesh: The mesh (of type open3d.geometry.TriangleMesh or similar) to be added to the visualizer.
        """
        self.vis.add_geometry(mesh)  # Add the mesh to the visualizer
        self.meshes.append(mesh)     # Store the mesh in the list of meshes
        
        

    def clear_meshes(self) -> None:
        """
        Remove all meshes from the visualizer and clear the list of meshes.
        """
        for mesh in self.meshes:             # Iterate over each mesh in the list
            self.vis.remove_geometry(mesh)   # Remove the mesh from the visualizer
        self.meshes = []      




@jit(nopython=True)
def _get_largest_cluster_label(labels: np.ndarray) -> int:
    """
    Get the label of the largest cluster from the given labels.

    Parameters:
    - labels: The labels assigned to each point by a clustering algorithm.

    Returns:
    - largest_cluster_label: The label of the largest cluster.
    """
    unique_labels = np.unique(labels)
    
    # Manually compute counts for each unique label
    counts = np.zeros(unique_labels.shape[0], dtype=np.int64)
    for i, label in enumerate(unique_labels):
        counts[i] = np.sum(labels == label)
    
    mask = unique_labels != -1
    unique_labels = unique_labels[mask]
    counts = counts[mask]
    
    largest_cluster_label = unique_labels[np.argmax(counts)]
    return largest_cluster_label

def segment_largest_cluster(pcl: o3d.geometry.PointCloud, eps: float = 0.05, min_samples: int = 10, max_points: int = 2000) -> o3d.geometry.PointCloud:
    """
    Segment the largest cluster from a point cloud using DBSCAN.

    Parameters:
    - pcl: The input point cloud.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    - max_points: Maximum number of points to consider. If the point cloud has more points, a random subset is taken.

    Returns:
    - largest_cluster_pcl: Point cloud of the largest cluster.
    """
    points = np.asarray(pcl.points)
    
    if max_points is not None and len(points) > max_points:
        selected_indices = np.random.choice(len(points), max_points, replace=False)
        points = points[selected_indices]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    largest_cluster_label = _get_largest_cluster_label(labels)

    largest_cluster_points = points[labels == largest_cluster_label]

    largest_cluster_pcl = o3d.geometry.PointCloud()
    largest_cluster_pcl.points = o3d.utility.Vector3dVector(largest_cluster_points)

    return largest_cluster_pcl


class HostSync:
    """
    HostSync is a utility class designed to synchronize multiple message streams based on their sequence numbers.
    It ensures that for a given sequence number, messages from all streams are present, allowing for synchronized
    processing of data from different sources.
    """

    def __init__(self) -> None:
        """
        Initialize the HostSync object with an empty dictionary to store arrays of messages.
        """
        self.arrays = {}

    def add_msg(self, name: str, msg: Any) -> Union[Dict[str, Any], bool]:
        """
        Add a message to the corresponding array based on its name.

        Parameters:
        - name: The name (or type) of the message.
        - msg: The message object to be added.

        Returns:
        - A dictionary of synced messages if there are enough synced messages, otherwise False.
        """
        # Check if the message type exists in the dictionary, if not, create an empty list for it
        if not name in self.arrays:
            self.arrays[name] = []

        # Append the message to the corresponding list along with its sequence number
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        # Check for messages with the same sequence number across all message types
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break

        # If there are 3 synced messages (in this use case color, depth, nn but you may add more by changing 3 to something else), 
        # remove all older messages and return the synced messages
        if len(synced) == 3:
            # Remove older messages with sequence numbers less than the current message's sequence number
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj["seq"] < msg.getSequenceNum():
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False

"""
shape_dict provides a mapping between object names and their corresponding geometric shapes. 
The shapes are approximated using basic geometric primitives. 
For objects that do not have a clear or simple geometric representation, the value is set to None.

Key: Object name (e.g., "person", "car", "apple"), must be a YOLO label
Value: Geometric shape representing the object "cylinder", "cuboid", "sphere", "plane" or None if no simple shape can represent the object.
"""
shape_dict = {
    "person": None,
    "bicycle": None,
    "car": "cuboid",
    "motorbike": None,
    "aeroplane": None,
    "bus": "cuboid",
    "train": "cuboid",
    "truck": "cuboid",
    "boat": "cuboid",
    "traffic light": None,
    "fire hydrant": None,
    "stop sign": "plane",
    "parking meter": None,
    "bench": "cuboid",
    "bird": None,
    "cat": None,
    "dog": None,
    "horse": None,
    "sheep": None,
    "cow": None,
    "elephant": None,
    "bear": None,
    "zebra": None,
    "giraffe": None,
    "backpack": "cuboid",
    "umbrella": None,
    "handbag": "cuboid",
    "tie": None,
    "suitcase": "cuboid",
    "frisbee": None,
    "skis": "plane",
    "snowboard": "plane",
    "sports ball": "sphere",
    "kite": None,
    "baseball bat": None,
    "baseball glove": None,
    "skateboard": "plane",
    "surfboard": "plane",
    "tennis racket": "plane",
    "bottle": None,
    "wine glass": None,
    "cup": None,
    "fork": None,
    "knife": None,
    "spoon": None,
    "bowl": None,
    "banana": None,
    "apple": "sphere",
    "sandwich": None,
    "orange": "sphere",
    "broccoli": None,
    "carrot": None,
    "hot dog": None,
    "pizza": "cuboid",
    "donut": None,
    "cake": "cuboid",
    "chair": "cuboid",
    "sofa": "cuboid",
    "pottedplant": "cuboid",
    "bed": "cuboid",
    "diningtable": "cuboid",
    "toilet": None,
    "tvmonitor": "cuboid",
    "laptop": "cuboid",
    "mouse": None,
    "remote": "cuboid",
    "keyboard": "cuboid",
    "cell phone": "cuboid",
    "microwave": "cuboid",
    "oven": "cuboid",
    "toaster": "cuboid",
    "sink": None,
    "refrigerator": "cuboid",
    "book": "cuboid",
    "clock": None,
    "vase": None,
    "scissors": None,
    "teddy bear": None,
    "hair drier": None,
    "toothbrush": None
}



def fit_and_create_sphere_mesh(points: np.ndarray, thresh: float = 0.2, verbose: bool = False) -> o3d.geometry.TriangleMesh:
    """
    fit_and_create_sphere_mesh fits a sphere to a set of 3D points using the RANSAC algorithm and then creates a mesh representation of the fitted sphere.

    Parameters:
    - points (numpy.ndarray): A set of 3D coordinates, where each point is represented as an array of [x, y, z] values.
    - thresh (float, optional): A threshold value for the RANSAC algorithm to determine inliers and outliers. Defaults to 0.2
    - verbose (bool, optional): If set to True, prints out the sphere parameters. Defaults to False.

    Returns:
    - sphere_mesh (o3d.geometry.TriangleMesh): A mesh representation of the fitted sphere.
    """
  
    # Fit a sphere using RANSAC
    sph = pyrsc.Sphere()
    center, radius, _ = sph.fit(points, thresh)
    
    # Print sphere parameters if verbose is True
    if verbose:
        print(f"Detected sphere with radius {radius:.2f} at location ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # Create a sphere mesh using Open3D
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_mesh.translate(center)
    
    return sphere_mesh

@jit(nopython=True)
def _project_points_onto_plane(points, best_eq):
    """
    Project a set of 3D points onto a plane.

    Parameters:
    - points: np.ndarray
        The 3D points to be projected.
    - best_eq: List[float]
        The equation of the plane [a, b, c, d] such that ax + by + cz + d = 0.

    Returns:
    - projected_points: np.ndarray
        The projected points onto the plane.
    """
    projected_points = np.empty(points.shape)
    for i, point in enumerate(points):
        t = -(best_eq[0] * point[0] + best_eq[1] * point[1] + best_eq[2] * point[2] + best_eq[3]) / (best_eq[0]**2 + best_eq[1]**2 + best_eq[2]**2)
        projected_point = point + t * np.array(best_eq[:3])
        projected_points[i] = projected_point
    return projected_points

@jit(nopython=True)
def _map_3d_to_2d(points, basis1, basis2):
    """
    Map 3D points to a 2D coordinate system defined by two basis vectors.

    Parameters:
    - points: np.ndarray
        The 3D points to be mapped.
    - basis1: np.ndarray
        The first basis vector of the 2D coordinate system.
    - basis2: np.ndarray
        The second basis vector of the 2D coordinate system.

    Returns:
    - points_2d: np.ndarray
        The mapped 2D points.
    """
    points_2d = np.empty((points.shape[0], 2))
    for i, point in enumerate(points):
        coord1 = np.dot(point, basis1)
        coord2 = np.dot(point, basis2)
        points_2d[i] = [coord1, coord2]
    return points_2d

@jit(nopython=True)
def _create_triangles_from_hull(hull_vertices_count):
    """
    Create triangles by connecting adjacent vertices of a convex hull.

    Parameters:
    - hull_vertices_count: int
        The number of vertices in the convex hull.

    Returns:
    - triangles: np.ndarray
        The triangles formed by connecting adjacent vertices of the convex hull.
    """
    triangles = np.empty((hull_vertices_count - 2, 3), dtype=np.int32)
    for i in range(1, hull_vertices_count - 1):
        triangles[i-1] = [0, i, i + 1]
    return triangles



def fit_and_create_plane_mesh(points: np.ndarray, thresh: float = 0.1, verbose: bool = False) -> o3d.geometry.TriangleMesh:
    """
    Fit a plane to a set of 3D points using the RANSAC algorithm and then creates a mesh representation of the fitted plane.

    Parameters:
    - points (numpy.ndarray): A set of 3D coordinates, where each point is an array with [x, y, z] values.
    - thresh (float, optional): A threshold value for the RANSAC algorithm to determine inliers and outliers. Defaults to 0.1.
    - verbose (bool, optional): If True, prints out the plane parameters. Defaults to False.

    Returns:
    - mesh (o3d.geometry.TriangleMesh): A mesh representation of the fitted plane.
    """    
    
    # Fit a plane using RANSAC
    plane = pyrsc.Plane()
    best_eq, inliers = plane.fit(points, thresh)
    
    # If verbose is True, print the plane parameters
    if verbose:
        a, b, c, d = best_eq
        print(f"Detected plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0.")
    
    # Project the inlier points onto the plane
    projected_points = _project_points_onto_plane(points[inliers], best_eq)
    
    # Define a 2D coordinate system on the plane
    normal = np.array(best_eq[:3])
    basis1 = np.cross(normal, normal + np.array([1, 0, 0]))
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    basis2 /= np.linalg.norm(basis2)
    
    # Map the 3D points to the 2D coordinate system
    points_2d = _map_3d_to_2d(projected_points, basis1, basis2)
    
    # Compute the 2D convex hull
    hull_2d = ConvexHull(points_2d)

    # Map the 2D convex hull back to 3D space
    hull_3d_points = []
    for vertex in hull_2d.vertices:
        point_2d = points_2d[vertex]
        point_3d = point_2d[0] * basis1 + point_2d[1] * basis2
        hull_3d_points.append(point_3d)
    hull_3d_points = np.array(hull_3d_points)

    # Create a mesh from the 3D convex hull points
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(hull_3d_points)

    # Create triangles by connecting adjacent vertices of the convex hull
    triangles = _create_triangles_from_hull(len(hull_3d_points))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute normals for shading
    mesh.compute_vertex_normals()
    
    return mesh


def fit_and_create_cuboid_mesh(points: np.ndarray, thresh: float = 0.01, max_iterations: int = 1000, verbose: bool = False) -> o3d.geometry.TriangleMesh:
    """
    Fit a cuboid to a set of 3D points using the RANSAC algorithm and then creates a mesh representation of the fitted cuboid.

    Parameters:
    - points (numpy.ndarray): A set of 3D coordinates, where each point is an array with [x, y, z] values.
    - thresh (float, optional): A threshold value for the RANSAC algorithm to determine inliers and outliers. Defaults to 0.01.
    - max_iterations (int, optional): Maximum number of iterations for the RANSAC algorithm. Defaults to 1000.
    - verbose (bool, optional): If True, prints out the cuboid parameters. Defaults to False.

    Returns:
    - cuboid_mesh (o3d.geometry.TriangleMesh): A mesh representation of the fitted cuboid.
    """
    
    # Fit a cuboid using RANSAC
    cuboid = pyrsc.Cuboid()
    cuboid_params = cuboid.fit(points, thresh, max_iterations)
    
    # Extract the detected planes and inliers
    planes = cuboid_params[0]
    inlier_indices = cuboid_params[1]
    
    # Extract the actual inlier points from the original point cloud
    inlier_points = points[inlier_indices]
    
    # Get parallel planes for each detected plane
    all_planes = []
    for plane in planes:
        parallel_planes = get_parallel_planes(plane, inlier_points)
        all_planes.extend(parallel_planes)
    
    # Generate the cuboid mesh from the planes
    cuboid_mesh = cuboid_from_planes(all_planes)
    
    # Compute the cuboid parameters
    min_point = np.min(inlier_points, axis=0)
    max_point = np.max(inlier_points, axis=0)
    midpoint = (min_point + max_point) / 2
    width = max_point[0] - min_point[0]
    height = max_point[1] - min_point[1]
    depth = max_point[2] - min_point[2]
    
    # If verbose is True, print the cuboid parameters in a single line
    if verbose:
        print(f"Detected cuboid with Midpoint: [{midpoint[0]:.2f}, {midpoint[1]:.2f}, {midpoint[2]:.2f}], Width: {width:.2f}, Height: {height:.2f}, Depth: {depth:.2f}")
    
    return cuboid_mesh


@jit(nopython=True)
def _project_onto_normal(inliers: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Project the inliers onto the plane's normal.

    Parameters:
    - inliers (numpy.ndarray): The inlier points, where each point is an array with [x, y, z] values.
    - normal (numpy.ndarray): The normal of the plane.

    Returns:
    - projections (numpy.ndarray): The projections of the inliers onto the plane's normal.
    """
    return np.dot(inliers, normal)


def get_parallel_planes(plane: List[float], inliers: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Given a plane and inliers, find the two parallel planes that bound the inliers.

    Parameters:
    - plane (List[float]): The equation of the plane [a, b, c, d] such that ax + by + cz + d = 0.
    - inliers (numpy.ndarray): The inlier points, where each point is an array with [x, y, z] values.

    Returns:
    - parallel_planes (Tuple[List[float], List[float]]): The equations of the two parallel planes.
    """
    normal = np.array(plane[:3])
    
    # Project the inliers onto the plane's normal
    projections = _project_onto_normal(inliers, normal)
    
    # Adjust the d value based on the min and max projections
    d_min = -np.min(projections)
    d_max = -np.max(projections)
    
    parallel_plane_1 = plane.copy()
    parallel_plane_1[3] = d_min

    parallel_plane_2 = plane.copy()
    parallel_plane_2[3] = d_max

    return parallel_plane_1, parallel_plane_2


@jit(nopython=True)
def _solve_system_of_equations(normals: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """
    Solve a system of linear equations to find the intersection point of three planes.

    Parameters:
    - normals (numpy.ndarray): The normals of the three planes.
    - constants (numpy.ndarray): The constants from the plane equations.

    Returns:
    - point (numpy.ndarray): The intersection point of the three planes, represented as [x, y, z].
    """
    return np.linalg.solve(normals, constants)


def intersect_planes(plane1: List[float], plane2: List[float], plane3: List[float]) -> np.ndarray:
    """
    Find the intersection point of three planes.

    Parameters:
    - plane1, plane2, plane3 (List[float]): The equations of the planes [a, b, c, d] such that ax + by + cz + d = 0.

    Returns:
    - point (numpy.ndarray): The intersection point of the three planes, represented as [x, y, z].
    """
    normals = np.array([plane1[:3], plane2[:3], plane3[:3]])
    constants = -np.array([plane1[3], plane2[3], plane3[3]])
    point = _solve_system_of_equations(normals, constants)
    return point



def create_quad_face(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Create a mesh from four vertices.

    Parameters:
    - p1, p2, p3, p4 (numpy.ndarray): Vertices of the quad face [x, y, z]

    Returns:
    - mesh (o3d.geometry.TriangleMesh): The quad mesh created from the four vertices.
    """
    
    # Combine the four vertices into a list
    vertices = [p1, p2, p3, p4]
    
    # Define the triangles that make up the quad face. 
    # The quad is split into two triangles for rendering.
    triangles = [[0, 1, 2], [1, 2, 3]]

    # Create a new TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    
    # Set the vertices of the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Set the triangles of the mesh
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    return mesh


def cuboid_from_planes(planes: List[List[float]], color: List[float] = [0.5, 0.5, 0.5]) -> o3d.geometry.TriangleMesh:
    """
    Generate a cuboid mesh from 6 plane equations.

    Parameters:
    planes (List[List[float]]): A list of 6 plane equations [a, b, c, d] such that ax + by + cz + d = 0.

    Returns:
    combined_mesh (o3d.geometry.TriangleMesh): An open3d.geometry.TriangleMesh object representing the cuboid.
    """
    
    # Generate all combinations of 5 planes
    combinations_of_five = list(combinations(planes, 5))

    quads = []
    for combination_of_five in combinations_of_five:
        combinations_of_three = list(combinations(combination_of_five, 3))
        vertices = []
        for combination in combinations_of_three:
            # Check if any two planes share a normal
            normals = [plane[:3] for plane in combination]
            if len(set(tuple(normal) for normal in normals)) < 3:
                continue  # Skip this combination if any two normals are the same
            vertices.append(intersect_planes(*combination))
        quads.append(vertices)

    meshes = []
    # Create a mesh for each quadrilateral and add it to the list
    for quad in quads:
        p1, p2, p3, p4 = quad
        quad_face = create_quad_face(p1, p2, p3, p4)
        meshes.append(quad_face)
    
    # Combine into single triangle mesh
    combined_mesh = meshes[0]
    for mesh in meshes[1:]:
        combined_mesh += mesh
    
    # Color the mesh
    combined_mesh.paint_uniform_color(color)

    return combined_mesh


def fit_and_create_cylinder_mesh(points: np.ndarray, thresh: float = 0.2, verbose: bool = False) -> o3d.geometry.TriangleMesh:
    """
    Fit a cylinder to a set of 3D points using the RANSAC algorithm and then creates a mesh representation of the fitted cylinder.

    Parameters:
    - points (numpy.ndarray): A set of 3D coordinates, where each point is an array with [x, y, z] values.
    - thresh (float, optional): A threshold value for the RANSAC algorithm to determine inliers and outliers. Defaults to 0.2.
    - verbose (bool, optional): If True, prints out the cylinder parameters. Defaults to False.

    Returns:
    - mesh_cylinder (o3d.geometry.TriangleMesh): A mesh representation of the fitted cylinder.
    """
    
    # Fit a cylinder using RANSAC
    cylinder = pyrsc.Cylinder()
    center, axis, radius, inliers = cylinder.fit(points, thresh)
    
    # Calculate the top and bottom endpoints of the cylinder
    top_point, bottom_point = cylinder_endpoints(center, axis, points[inliers])
    
    # Create a cylinder mesh
    height = np.linalg.norm(top_point - bottom_point)
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    
    # Translate the cylinder to the midpoint of top and bottom points
    midpoint = (top_point + bottom_point) / 2
    mesh_cylinder.translate(midpoint)
    
    # Rotate the cylinder to align with the given axis
    z_axis = np.array([0, 0, 1])
    rotation_matrix = get_rotation_matrix_from_vectors(z_axis, axis)
    mesh_cylinder.rotate(rotation_matrix, center=midpoint)
    
    # If verbose is True, print the cylinder parameters in a single line
    if verbose:
        print(f"Detected cylinder with Radius: {radius:.2f}, Height: {height:.2f}, Midpoint: [{midpoint[0]:.2f}, {midpoint[1]:.2f}, {midpoint[2]:.2f}], Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")
    
    return mesh_cylinder


@jit(nopython=True)
def _compute_rotation_matrix(v: np.ndarray, s: float, c: float) -> np.ndarray:
    """
    Compute the rotation matrix using the Rodriguez rotation formula.

    Parameters:
    - v: Cross product of vectors 'a' and 'b'.
    - s: Sine of the angle between vectors 'a' and 'b'.
    - c: Cosine of the angle between vectors 'a' and 'b'.

    Returns:
    - rotation_matrix: The 3x3 rotation matrix.
    """
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_matrix = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / s**2)
    return rotation_matrix

def get_rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix that aligns vector 'a' to vector 'b'.

    Parameters:
    - a: The source vector that needs to be rotated.
    - b: The target vector to which 'a' should be aligned.

    Returns:
    - rotation_matrix: The 3x3 rotation matrix that rotates 'a' to align with 'b'.
    """
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    return _compute_rotation_matrix(v, s, c)


@jit(nopython=True)
def _compute_endpoints(center: np.ndarray, axis: np.ndarray, inlier_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the top and bottom endpoints of a cylinder.

    Parameters:
    - center: The center point of the cylinder.
    - axis: The axis of the cylinder.
    - inlier_points: The inlier points that lie on the surface of the cylinder.

    Returns:
    - top_point: The top endpoint of the cylinder.
    - bottom_point: The bottom endpoint of the cylinder.
    """
    t_values = np.array([(np.dot(point - center, axis)) for point in inlier_points])
    top_t = np.max(t_values)
    bottom_t = np.min(t_values)
    top_point = center + top_t * axis
    bottom_point = center + bottom_t * axis
    return top_point, bottom_point


def cylinder_endpoints(center: np.ndarray, axis: np.ndarray, inlier_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the top and bottom endpoints of a cylinder given its center, axis, and inlier points.

    Parameters:
    - center: The center point of the cylinder.
    - axis: The axis of the cylinder.
    - inlier_points: The inlier points that lie on the surface of the cylinder.

    Returns:
    - top_point: The top endpoint of the cylinder.
    - bottom_point: The bottom endpoint of the cylinder.
    """
    axis = axis / np.linalg.norm(axis)
    return _compute_endpoints(center, axis, inlier_points)





def fit_shape_to_detection(detection: str, point_cloud: o3d.geometry.PointCloud, shape_dict: dict, verbose: bool = False) -> Union[o3d.geometry.TriangleMesh, None]:
    """
    Fit a geometric shape to a given detection based on a predefined dictionary of shapes. 
    The function selects the appropriate shape fitting method based on the detected object class.

    Parameters:
    - detection (str): The object class detected, e.g., "car", "person", etc.
    - point_cloud (o3d.geometry.PointCloud): The 3D point cloud data associated with the detection.
    - shape_dict (dict): A dictionary mapping object classes to their corresponding geometric shapes.
    - verbose (bool, optional): If True, prints out the shape parameters. Defaults to False.

    Returns:
    - o3d.geometry.TriangleMesh or None: The fitted mesh for the detection. If the shape type is not defined or there are insufficient points, it returns None.
    """
    
    # Convert the point cloud to a numpy array
    points = np.asarray(point_cloud.points)
    
    # Check if there are enough points for shape fitting
    if len(points) < 20:
        print("Not enough points for shape fitting.")
        return None
    
    # If there are too many points, randomly sample a subset for efficiency
    if len(points) > 300:
        selected_indices = np.random.choice(len(points), 300, replace=False)
        points = points[selected_indices]

    # Get the shape type for the detected object class from the dictionary
    shape_type = shape_dict.get(detection, None)
    
    # If the shape type is not defined, return None
    if shape_type is None:
        return None
    
    # Fit the appropriate shape based on the shape type
    if shape_type == "sphere":
        mesh = fit_and_create_sphere_mesh(points, verbose=verbose)
    elif shape_type == "plane":
        mesh = fit_and_create_plane_mesh(points, verbose=verbose)
    elif shape_type == "cylinder":
        mesh = fit_and_create_cylinder_mesh(points, verbose=verbose)
    elif shape_type == "cuboid":
        mesh = fit_and_create_cuboid_mesh(points, verbose=verbose)
    else:
        # Raise an error if the shape type is unknown
        raise ValueError(f"Unknown shape type: {shape_type}")
        
    return mesh


# selected model and its config (YOLO v5)
args = {
    'model': 'yolov5n_coco_416x416',
    'config': 'json/yolov5.json'
}

# parse config
configPath = Path(args["config"])
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args["model"]
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args["model"], shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Initialize the pipeline
pipeline = dai.Pipeline()

# Create nodes for the left and right cameras
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)

# Create a node for the stereo depth calculation
stereo = pipeline.create(dai.node.StereoDepth)

# Create a node for the central RGB camera
camRgb = pipeline.create(dai.node.ColorCamera)

# Create NeuralNetwork node and set the YOLOv4 model
yolo_nn = pipeline.create(dai.node.YoloDetectionNetwork)
yolo_nn.setConfidenceThreshold(confidenceThreshold)
yolo_nn.setNumClasses(classes)
yolo_nn.setCoordinateSize(coordinates)
yolo_nn.setAnchors(anchors)
yolo_nn.setAnchorMasks(anchorMasks)
yolo_nn.setIouThreshold(iouThreshold)
yolo_nn.setBlobPath(nnPath)
yolo_nn.setNumInferenceThreads(2)
yolo_nn.input.setBlocking(False)

# Configure the MonoCamera nodes
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Link the left and right cameras to the stereo depth node
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create an output node for the depth stream
depth_out = pipeline.create(dai.node.XLinkOut)
depth_out.setStreamName("depth")
stereo.depth.link(depth_out.input)

# Configure the camRgb node
camRgb_out = pipeline.create(dai.node.XLinkOut)
camRgb_out.setStreamName("color")
camRgb.video.link(camRgb_out.input)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

# Link the central RGB camera output to the YoloDetectionNetwork node
camRgb.preview.link(yolo_nn.input)

# Create an output node for the YOLO detections
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
yolo_nn.out.link(xout_nn.input)




class FrequencyMeter:
    """
    A utility class to measure the frequency of events over a specified duration.

    Attributes:
    - timestamps (List[float]): A list of timestamps when the events occurred.
    - avg_over (int): The number of events over which the frequency is averaged.

    Methods:
    - tick(): Records the current timestamp and calculates the average frequency over the last 'avg_over' events.
    """

    def __init__(self, avg_over=10):
        """
        Initializes the FrequencyMeter with a specified number of events to average over.

        Parameters:
        - avg_over (int, optional): The number of events over which the frequency is averaged. Defaults to 10.
        """
        self.timestamps = []
        self.avg_over = avg_over

    def tick(self):
        """
        Records the current timestamp and calculates the average frequency over the last 'avg_over' events.
        If the number of recorded events exceeds 'avg_over', the oldest timestamp is removed.
        The calculated frequency is printed to the console.
        """
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_over:
            self.timestamps.pop(0)
            freq = self.avg_over / (self.timestamps[-1] - self.timestamps[0])
            print(f"Frequency: {freq:.2f} Hz")

freq_meter = FrequencyMeter()



# Set this to True if you want to print detections
verbose=False


with dai.Device(pipeline) as device:
    
    # Normalize bounding box coordinates to match frame dimensions
    def frameNorm(frame: np.ndarray, bbox: List[float]) -> List[int]:
        """
        Normalize bounding box coordinates based on frame dimensions.

        Parameters:
        - frame (np.ndarray): The frame to which the bounding box corresponds.
        - bbox (List[float]): Bounding box coordinates in the range <0..1>.

        Returns:
        - List[int]: Normalized bounding box coordinates in pixels.
        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name: str, frame: np.ndarray, detections: List[dai.ImgDetection]):
        """
        Display the frame with bounding boxes and labels.

        Parameters:
        - name (str): Window name for displaying the frame.
        - frame (np.ndarray): The frame to display.
        - detections (List[dai.ImgDetection]): List of detections to be drawn on the frame.
        """


        color = (255, 0, 0) # Bounding box color
        # Add bounding boxes and text onto the rgb frame according to detections
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Display the frame
        cv2.imshow(name, frame)

    # Set the brightness of the IR laser dot projector
    device.setIrLaserDotProjectorBrightness(1200)
    
    # Create output queues for depth, color, and neural network outputs
    qs = []
    qs.append(device.getOutputQueue("depth", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("color", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("nn", maxSize=1, blocking=False))

    # Retrieve camera calibration data
    calibData = device.readCalibration()
    w, h = camRgb.getIspSize()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, dai.Size2f(w, h))
    
    # Initialize the PointCloudVisualizer with the camera intrinsics
    pcl_converter = PointCloudVisualizer(intrinsics, w, h)

    # Initialize the HostSync class for synchronizing messages
    sync = HostSync()

    while True:
        # Iterate over each queue (depth, color, nn)
        for q in qs:
            # Try to get a new message from the queue
            new_msg = q.tryGet()
            if new_msg is not None:
                # Add the new message to the synchronized message list
                msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:
                    # Extract depth frame, color frame, and detections from the synchronized messages
                    depth = msgs["depth"].getFrame()
                    color = msgs["color"].getCvFrame()
                    detections = msgs["nn"].detections

                    # Convert the color frame from BGR to RGB and resize it to match the depth frame dimensions
                    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    rgb_resized = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

                    # Initialize an empty list to store all meshes
                    all_meshes = []

                    # Clear any existing meshes in the point cloud converter
                    pcl_converter.clear_meshes()

                    # Process each detection
                    for detection in detections:
                        # Extract and normalize the bounding box coordinates
                        xmin, ymin, xmax, ymax = [
                            int(detection.xmin * w),
                            int(detection.ymin * h),
                            int(detection.xmax * w),
                            int(detection.ymax * h),
                        ]

                        # Ensure bounding box coordinates are valid
                        xmin, xmax = sorted([xmin, xmax])
                        ymin, ymax = sorted([ymin, ymax])
                        xmin = max(0, min(xmin, rgb_resized.shape[1] - 1))
                        ymin = max(0, min(ymin, rgb_resized.shape[0] - 1))
                        xmax = min(rgb_resized.shape[1] - 1, max(xmax, 0))
                        ymax = min(rgb_resized.shape[0] - 1, max(ymax, 0))

                        # Skip invalid bounding boxes
                        if xmin >= xmax or ymin >= ymax:
                            print("Invalid bounding box detected. Skipping this detection.")
                            continue

                        # Create a modified depth map with values outside the bounding box set to zero
                        modified_depth = np.copy(depth)
                        modified_depth[0:ymin, :] = 0
                        modified_depth[ymax:, :] = 0
                        modified_depth[:, 0:xmin] = 0
                        modified_depth[:, xmax:] = 0

                        # Convert the modified depth and RGB data to a point cloud
                        object_pcl = pcl_converter.rgbd_to_projection(modified_depth, rgb_resized)

                        # Segment the largest cluster from the point cloud
                        object_pcl = segment_largest_cluster(object_pcl)

                        # Fit a shape to the detection and obtain the corresponding mesh
                        mesh = fit_shape_to_detection(labels[detection.label], object_pcl, shape_dict, verbose=verbose)

                        # If a valid mesh is obtained, add it to the point cloud converter
                        if mesh is not None:
                            pcl_converter.add_mesh(mesh)

                    # Convert the entire depth and RGB data to a point cloud
                    pcd = pcl_converter.rgbd_to_projection(depth, rgb_resized)

                    # Display the color frame with bounding boxes and labels
                    displayFrame("rgb", color, detections)

                    # Visualize the point cloud
                    pcl_converter.visualize_pcd()
                    
                    # Tick the framerate meter
                    freq_meter.tick()

        # Check for user input to exit the loop
        key = cv2.waitKey(1)
        if key == ord("q"):
            pcl_converter.close_window()
            cv2.destroyAllWindows()
            break