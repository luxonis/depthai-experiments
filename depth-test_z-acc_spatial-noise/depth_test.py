import open3d as o3d
import numpy as np
from utils import *
import math
from camera import Camera
from oak_camera import OakCamera

class DepthTest:
	def __init__(self):
		self.camera_dir = np.array([0, 0, -1])
		self.plane_normal = np.array([0, 0, -1])
		self.camera_wall_distance = 1 # m
		self.plane_distance = -self.camera_wall_distance
		self.plane_coeffs = (0,0,-self.camera_wall_distance)

		self.point_cloud_corrected = o3d.geometry.PointCloud()
		self.plane_fit_corrected_pcl = o3d.geometry.PointCloud()
		self.plane_fit_pcl = o3d.geometry.PointCloud()
		self.plane_fit_visualization = False

		self.fitted = False

		self.z_accuracy_medians = []
		self.z_means = []
		self.spatial_noise_rmses = []
		self.subpixel_spatial_noise_rmses = []
		self.samples = 0

	def set_ground_truth(self, point_cloud: o3d.geometry.PointCloud):
		points = np.asarray(point_cloud.points)
		Z = points[:, 2]

		self.camera_wall_distance = -np.median(Z)
		return self.camera_wall_distance

	def fit_plane(self, point_cloud: o3d.geometry.PointCloud):
		points = np.asarray(point_cloud.points)
		G = np.ones_like(points)
		G[:, 0] = points[:, 0]
		G[:, 1] = points[:, 1]
		Z = points[:, 2]
		(k_x, k_y, k_z), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
		normal = np.array([k_x, k_y, -1])
		normal = normal / np.linalg.norm(normal)

		self.plane_normal = normal
		self.plane_distance = k_z
		self.plane_coeffs = (k_x, k_y, k_z)

		self.fitted = True

		return normal, k_z

	def compute_tilt(self):
		horizontal_tilt = angle(self.camera_dir, self.plane_normal, np.array([0, 1, 0])) # in radians (- left, + right)
		vertical_tilt = angle(self.camera_dir, self.plane_normal, np.array([1, 0, 0])) # in radians (- up, + down)

		return horizontal_tilt, vertical_tilt

	def print_tilt(self):
		horizontal_tilt, vertical_tilt = self.compute_tilt()
		print(
			f"Horizontal tilt: {abs(horizontal_tilt) * 180/np.pi}° {'LEFT' if horizontal_tilt < 0 else 'RIGHT'}", 
			f"Vertical tilt: {abs(vertical_tilt) * 180/np.pi}° {'UP' if vertical_tilt < 0 else 'DOWN'}"
		)

	def project_on_plane(self, point_cloud: o3d.geometry.PointCloud):
		"""
		Project the `point_cloud` on the fitted plane **along the z-axis** and return the projected point cloud.

		The only appropriate use of this function is for visualization purposes.
		"""
		points = np.asarray(point_cloud.points)
		G = np.ones_like(points)
		G[:, 0] = points[:, 0]
		G[:, 1] = points[:, 1]
		plane_fit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.hstack([points[:,0:2], G @ np.matrix(self.plane_coeffs).T])))
		return plane_fit

	def correct_tilt(self, point_cloud: o3d.geometry.PointCloud):
		# Correct the point cloud tilt (isolating camera positioning errors)
		# The plane is corrected with angle (acos(camera_dir @ normal)) and axis (np.cross(camera_dir, normal)) instad of the horizontal and vertical tilt to reduce numerical errors
		# R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([-vertical_tilt, -horizontal_tilt, 0]))
		axis_angle = normalize(np.cross(self.plane_normal, self.camera_dir)) * np.arccos(np.dot(self.camera_dir, self.plane_normal))
		R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
		point_cloud_corrected = o3d.geometry.PointCloud(point_cloud)
		point_cloud_corrected.rotate(R, center=(0, 0, 0))
		return point_cloud_corrected

	def measure(self, camera: Camera):
		point_cloud = camera.point_cloud
		print(f"\rAdding measurements {'.'*self.samples}", end="")

		point_cloud_corrected = self.correct_tilt(point_cloud)

		spatial_noise = self.compute_spatial_noise(point_cloud_corrected)
		self.spatial_noise_rmses.append(spatial_noise)

		if isinstance(camera, OakCamera):
			subpixel_spatial_noise = self.compute_subpixel_spatial_noise(
				point_cloud_corrected, 
				focal_length=camera.focal_length,
				stereoscopic_baseline=camera.stereoscopic_baseline
			)
			self.subpixel_spatial_noise_rmses.append(subpixel_spatial_noise)

		z_accuracy = self.compute_z_accuracy(point_cloud_corrected)
		self.z_accuracy_medians.append(z_accuracy)

		self.samples += 1

		return spatial_noise, subpixel_spatial_noise, z_accuracy

	def compute_spatial_noise(self, point_cloud_corrected: o3d.geometry.PointCloud):
		point_cloud_corrected_points = np.asarray(point_cloud_corrected.points)

		z_error = -point_cloud_corrected_points[:,2] - self.camera_wall_distance
		z_error = np.sort(z_error)
		# remove values below 0.5% and above 99.5%
		z_error = z_error[int(len(z_error)*0.005):int(len(z_error)*0.995)]
		RMSE = np.sqrt(np.mean(z_error**2))

		return RMSE

	def compute_subpixel_spatial_noise(self, point_cloud_corrected: o3d.geometry.PointCloud, focal_length: float = math.nan, stereoscopic_baseline: float = math.nan):
		if math.isnan(focal_length) or math.isnan(stereoscopic_baseline):
			return math.nan
		
		point_cloud_corrected_points = np.asarray(point_cloud_corrected.points)
		z = -point_cloud_corrected_points[:,2]
		disparity = (stereoscopic_baseline * focal_length) / z
		disparity_ground_truth = (stereoscopic_baseline * focal_length) / self.camera_wall_distance
		disparity_error = disparity - disparity_ground_truth
		# remove values below 0.5% and above 99.5%
		disparity_error = disparity_error[int(len(disparity_error)*0.005):int(len(disparity_error)*0.995)]
		RMSE = np.sqrt(np.mean(disparity_error**2))

		return RMSE

	def compute_z_accuracy(self, point_cloud_corrected: o3d.geometry.PointCloud):
		point_cloud_corrected_points = np.asarray(point_cloud_corrected.points)
		self.z_means.append(np.mean(point_cloud_corrected_points[:,2]))

		z_error = -point_cloud_corrected_points[:, 2] - self.camera_wall_distance
		# remove values below 0.5% and above 99.5%
		z_error = np.sort(z_error)
		z_error = z_error[int(len(z_error)*0.005):int(len(z_error)*0.995)]
		median = np.median(z_error)

		return median

	def reset(self):
		self.z_accuracy_medians = []
		self.spatial_noise_rmses = []
		self.subpixel_spatial_noise_rmses = []
		self.z_means = []
		self.samples = 0

	def print_results(self):
		print("=== Results ===")
		print(f"{self.samples} measurements")
		print(f"Z accuracy: {np.mean(self.z_accuracy_medians) / self.camera_wall_distance * 100:.2f}% of GT (avg distance: {-np.mean(self.z_means)*1000:.2f}mm)")
		print(f"Spatial noise: {np.mean(self.spatial_noise_rmses)*1000:.2f} mm")
		print(f"Subpixel spatial noise: {np.mean(self.subpixel_spatial_noise_rmses):.2f} px")
		print()

	def show_plane_fit_visualization(self, point_cloud: o3d.geometry.PointCloud):
		if not self.fitted:
			print("Plane not fitted yet")
			return
		# red - original fitted plane
		# green - corrected fitted plane
		# colored - corrected point cloud
		point_cloud_corrected = self.correct_tilt(point_cloud)
		plane_fit = self.project_on_plane(point_cloud)
		plane_fit.paint_uniform_color([1, 0, 0])
		plane_fit_corrected = self.correct_tilt(plane_fit)
		plane_fit_corrected.paint_uniform_color([0, 1, 0])


		self.point_cloud_corrected.points = point_cloud_corrected.points
		self.plane_fit_corrected_pcl.points =  plane_fit_corrected.points
		self.plane_fit_pcl.points = plane_fit.points

		self.point_cloud_corrected.colors = point_cloud_corrected.colors
		self.plane_fit_corrected_pcl.colors =  plane_fit_corrected.colors
		self.plane_fit_pcl.colors = plane_fit.colors

		self.plane_fit_visualization = True

	def hide_plane_fit_visualization(self):
		empty_pcl = o3d.geometry.PointCloud()
		self.point_cloud_corrected.points = empty_pcl.points
		self.plane_fit_corrected_pcl.points =  empty_pcl.points
		self.plane_fit_pcl.points = empty_pcl.points

		self.point_cloud_corrected.colors = empty_pcl.colors
		self.plane_fit_corrected_pcl.colors =  empty_pcl.colors
		self.plane_fit_pcl.colors = empty_pcl.colors
		
		self.plane_fit_visualization = False



