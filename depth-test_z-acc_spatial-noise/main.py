import cv2
import open3d as o3d
import numpy as np
import depthai as dai
from camera import Camera
from typing import List
from utils import *
import config


camera = Camera(dai.DeviceInfo())

ROI = (0, 0, 0, 0)
ROI = (135, 64, 224, 220)
while True:
	key = cv2.waitKey(1)

	# QUIT - press the `q` key
	if key == ord('q'):
		break
	
	camera.update()
	if camera.image_frame is None:
		continue

	# SELECT ROI - press the `r` key
	if key == ord('r'):
		ROI = cv2.selectROI("image", camera.image_frame)
		print(ROI)

	ROI_mask = np.zeros_like(camera.image_frame)
	ROI_mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] = 1
	img = camera.image_frame*0.2 + (camera.image_frame * ROI_mask)*0.8
	cv2.imshow("image", img.astype(np.uint8))

	# SHOW THE POINT CLOUD - press the `p` key
	if key == ord('p'):
		print("=== TEST ===")

		# Get the point cloud
		ROI_mask = np.zeros_like(camera.depth_frame)
		ROI_mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] = 1
		depth = camera.depth_frame * ROI_mask
		point_cloud = camera.rgbd_to_point_cloud(depth, cv2.cvtColor(camera.image_frame, cv2.COLOR_RGB2BGR))

		# Fit a plane to the point cloud
		points = np.asarray(point_cloud.points)
		G = np.ones_like(points)
		G[:, 0] = points[:, 0]
		G[:, 1] = points[:, 1]
		Z = points[:, 2]
		(a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
		camera_dir = np.array([0, 0, -1])
		normal = np.array([a, b, -1])
		normal = normal / np.linalg.norm(normal)

		# Compute the horizontal and vertical tilt angles
		horizontal_tilt = angle(camera_dir, normal, np.array([0, 1, 0])) # in radians (- left, + right)
		vertical_tilt = angle(camera_dir, normal, np.array([1, 0, 0])) # in radians (- up, + down)

		print(
			f"Horizontal tilt: {abs(horizontal_tilt) * 180/np.pi}° {'LEFT' if horizontal_tilt < 0 else 'RIGHT'}", 
			f"Vertical tilt: {abs(vertical_tilt) * 180/np.pi}° {'UP' if vertical_tilt < 0 else 'DOWN'}"
		)

		origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=(0,0,-1))
		plane_fit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.hstack([points[:,0:2], G @ np.matrix([a, b, c]).T])))
		plane_fit.paint_uniform_color([0, 0, 0.9])

		plane_fit_corrected = o3d.geometry.PointCloud(plane_fit)
		plane_fit_corrected.paint_uniform_color(np.array([0,0.9,0]))
		point_cloud_corrected = o3d.geometry.PointCloud(point_cloud)

		# Correct the point cloud tilt (isolating camera positioning errors)
		# The plane is corrected with angle (acos(camera_dir @ normal)) and axis (np.cross(camera_dir, normal)) instad of the horizontal and vertical tilt to reduce numerical errors
		# R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([-vertical_tilt, -horizontal_tilt, 0]))
		axis_angle = normalize(np.cross(normal, camera_dir)) * np.arccos(np.dot(camera_dir, normal))
		R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
		point_cloud_corrected.rotate(R, center=(0,0,0))
		plane_fit_corrected.rotate(R, center=(0,0,0))


		plane_fit_corrected_points = np.asarray(plane_fit_corrected.points)
		point_cloud_corrected_points = np.asarray(point_cloud_corrected.points)
		print(f"[Z-values] avg: {np.average(plane_fit_corrected_points[:,2])} | min: {np.min(plane_fit_corrected_points[:,2])} | max: {np.max(plane_fit_corrected_points[:,2])} | std: {np.std(plane_fit_corrected_points[:,2])} | diff: {(np.max(plane_fit_corrected_points[:,2]) - np.min(plane_fit_corrected_points[:,2]))*1000}")
		
		# Compute Spatial noise
		z_error = plane_fit_corrected_points[:,2] - point_cloud_corrected_points[:,2]
		z_error = np.sort(z_error)
		# remove values below 0.5% and above 99.5%
		z_error = z_error[int(len(z_error)*0.005):int(len(z_error)*0.995)]
		RMSE = np.sqrt(np.mean(z_error**2))
		print(f"Spatial noise (RMSE): {RMSE*1000} mm")

		# Compute Z-Accuracy
		z_error = -plane_fit_corrected_points[:, 2] - config.camera_wall_distance
		# remove values below 0.5% and above 99.5%
		z_error = np.sort(z_error)
		z_error = z_error[int(len(z_error)*0.005):int(len(z_error)*0.995)]
		median = np.median(z_error)
		print(f"Z-Accuracy (median): {median*1000} mm")


		# Visualize the results
		o3d.visualization.draw_geometries([origin, point_cloud_corrected, plane_fit_corrected])