#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import cv2
import depthai
from projector_3d import PointCloudVisualizer
import numpy as np
from time import sleep
import open3d as o3d

# carry out unit tests
def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

curr_dir = str(Path('.').resolve().absolute())
is_color = True
final_path = curr_dir + '/dataset/left_1.png'
color = cv2.imread(final_path)
if is_color:
    final_path = curr_dir + '/dataset/color_1.png'
    color = cv2.imread(final_path)

final_path = curr_dir + '/dataset/right_1.png'
right = cv2.imread(final_path)

final_path = curr_dir + '/dataset/depth_1.tif'
depth = cv2.imread(final_path, cv2.IMREAD_UNCHANGED)
# sleep(2)
pixel_coords = pixel_coord_np(1280, 720) 

# Right camera            
M2 = np.array([[862.3698, 0.0,            637.838], 
               [0.0,          861.3336791992188, 364.46539306640625], 
               [0.0,          0.0,                   1.0]], dtype=np.float32)

with open(curr_dir + '/dataset/m2_r.npy', 'rb') as f:
    M2 = np.load(f)

print("Displaying sHARE OF COLOR IMAGE")
print(color.shape) 
print(depth.shape) 
print(depth.dtype)
print(depth.max())
print(depth.min())  

index = 2
print()
print()

# P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
# P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
# P3D.z = depth(x_d,y_d)

# M_RGB = np.array([[938.1863, 0, 659.42112274],
#                 [0, 936.9993429, 458.2857],
#                 [0,     0,          1   ]], dtype=np.float32)

# left camera
M_RGB = np.array([[858.826904296875, 0.0, 637.2894897460938], 
                  [0.0, 859.1322021484375, 319.0586853027344], 
                  [0.0, 0.0, 1.0]], dtype=np.float32)
if is_color:
    #color
    # M_RGB = np.array([[982.0190877, 0, 661.81781732],
    #                 [0, 980.77662276, 356.83627635],
    #                 [0,     0,          1   ]], dtype=np.float32)
    M_RGB = np.array([[997.95140865, 0, 636.89066558],
                      [0, 996.68878, 358.062592],
                      [0,     0,          1   ]], dtype=np.float32)

    M_RGB = np.array([[997.163574,    0.000000,  631.543762],
                    [0.000000,  997.899414,  365.938538],
                    [0.000000,    0.000000,    1.000000]], dtype=np.float32)

    with open(curr_dir + '/dataset/m3_rgb.npy', 'rb') as f:
        M_RGB = np.load(f)

    scale_width = 1280/1920
    m_scale = [[scale_width,      0,   0],
                [0,         scale_width,        0],
                [0,             0,         1]]
    M_RGB = np.matmul(m_scale, M_RGB)


R = np.array([[0.999966,    0.006664,   -0.004894],
              [-0.006641,    0.999967,    0.00470],
              [0.004925,   -0.004668,    0.999977]], dtype=np.float32)



if is_color:
    # color
    # R = np.array([[0.999684, 0.024849, -0.003688],
    #             [-0.024845,  0.999691,  0.001207],
    #             [0.003717,  -0.001115,  0.999992]], dtype=np.float32)
    # R = np.array([[1, 0., -0.0],
    #             [-0.,  1,  0.0],
    #             [0.,  -0.,  1]], dtype=np.float32)
    
    with open(curr_dir + '/dataset/rot_rgb.npy', 'rb') as f:
        R = np.load(f)

# T = np.array([-3.760219, 6.551057, -5.57906], np.float32)
# T = np.array([-3.760219, 6.551057, -5.57906], np.float32)
T = np.array([-7.5, 0, 0.0001], np.float32) # cm 
if is_color:
    # T = np.array([-3.770253, 0.055446, 0.251579], np.float32) # cm 
    with open(curr_dir + '/dataset/t_rgb.npy', 'rb') as f:
        T = np.load(f)

R_inv = np.linalg.inv(R)

T_neg = -1 * T

# H_inv = np.array([[ 1.00280449e+00, -7.35970674e-03, -2.78654202e+00],
#                   [ 6.93586028e-03,  9.99131674e-01, -2.22207977e+00],
#                   [ 4.35014077e-06, -2.74672190e-06,  9.98013113e-01]], dtype=np.float32)  

# H_inv = np.array([[ 9.93284397e-01,  1.68031060e-02,  5.75327534e+00],
#                     [-2.09308677e-02,  9.99638404e-01,  1.37869851e+01],
#                     [-1.02260173e-05, -5.87467424e-07,  1.00670350e+00]], dtype=np.float32)  

with open(curr_dir + '/dataset/inv_homo.npy', 'rb') as f:
    H_inv = np.load(f)
    H_inv = np.linalg.inv(H_inv)

im_color_right_unrec = (65535 // depth).astype(np.uint8)
# colorize depth map, comment out code below to obtain grayscale
im_color_right_unrec = cv2.applyColorMap(im_color_right_unrec, cv2.COLORMAP_HOT)
# cv2.imshow('color_map', im_color)

# backtorgb = cv2.cvtColor(depth_rgb, cv2.COLOR_GRAY2RGB)
# print(color.shape)
# print(backtorgb.shape)
added_image_right_unewq = cv2.addWeighted(right,0.6,im_color_right_unrec,0.3,0)
cv2.imshow('RGBD overlay right and depth raw', added_image_right_unewq)

# converting depth from rectified right to right frame_bgr
depth_vals = cv2.warpPerspective(depth, H_inv, depth.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

# depth_mat = depth_vals.copy()


im_color_right = (65535 // depth_vals).astype(np.uint8)
                            # colorize depth map, comment out code below to obtain grayscale
im_color_right = cv2.applyColorMap(im_color_right, cv2.COLORMAP_HOT)
# cv2.imshow('color_map', im_color)

# backtorgb = cv2.cvtColor(depth_rgb, cv2.COLOR_GRAY2RGB)
# print(color.shape)
# print(backtorgb.shape)
added_image_right = cv2.addWeighted(right,0.6,im_color_right,0.3,0)
cv2.imshow('RGBD overlay right and depth after removing rectified', added_image_right)
# cv2.waitKey()

# depth_vals = depth.copy()
            
## Projecting depth in right to world
K_inv = np.linalg.inv(M2) # right intriniscs inverse

print('Inv of intrinics of right camera')
print(K_inv)
print('pixel_coords.shape') # 
print(pixel_coords.shape) # (3,W x H) ->
print(pixel_coords[:,index])
print(depth_vals[3,1160])

temp = depth_vals.copy() # depth in right frame
x = temp.flatten()
print('flatten depth shape') #
print(x.shape)

cam_coords = np.dot(K_inv, pixel_coords) * temp.flatten() * 0.1 # [x, y, z]
del temp

print('cloud shape')
print(cam_coords.shape)
# print(len(cam_coords))
# x = np.ones_like(cam_coords[0])
# print(x.shape)

# pcd.crop()
cam_coords[:, cam_coords[2] > 500] = float('inf') 
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
pcd = o3d.geometry.PointCloud()
# vis = o3d.visualization.Visualizer()
# vis.create_window()

pcd.points = o3d.utility.Vector3dVector(cam_coords.transpose())
pcd.remove_non_finite_points()

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
# vis.add_geometry(pcd)
# vis.add_geometry(origin)
# # vis.update_renderer()
# print('x')
# while 1:
#     # vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()

print('printing single coordinate x , y, z')
print(cam_coords[:,index])
print('Printing M2 ---------------->')
print(M2)

print('Printing MRGB ---------------->')
print(M_RGB)

cam_coords_2 = np.vstack((cam_coords, np.ones_like(cam_coords[0]))) # [x,y,z,w]
print(cam_coords_2.shape)

print('printing single coordinate')
print(cam_coords_2[:,index])

extrensics = np.hstack((R, np.transpose([T])))

print('extrensics.shape')
print(extrensics.shape)
print(extrensics)
print('extrensics.shape using R inv and T neg inverse ')

extrensics = np.hstack((R_inv, np.transpose([T_neg])))
extrensics_2 = np.vstack((extrensics, np.array([0, 0, 0, 1])))

print(extrensics_2)
pcd.transform(extrensics_2)
# o3d.visualization.draw_geometries([pcd])

# vis.add_geometry(pcd)
# vis.add_geometry(origin)
# # vis.update_renderer()
# print('x')
# while 1:
#     # vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()



# [[ 9.999460e-01, -8.328000e-03, -3.621000e-03, -3.760219e+00],
#  [ 8.460000e-03,  9.998789e-01,  1.873400e-02,  5.105700e-02],
#  [ 3.463000e-03, -1.876400e-02,  9.998180e-01,  4.790600e-02]]
#inv
# [[ 9.9997097e-01,  8.3937785e-03,  3.4642764e-03,  3.7602191e+00],
#  [-8.3929347e-03,  9.9969912e-01, -1.8762169e-02, -5.1057000e-02],
#  [-3.6210436e-03,  1.8732697e-02,  9.9981791e-01, -4.7906000e-02]]

# rgb_frame_ref_cloud = np.matmul(extrensics, cam_coords_2)
# pcd.points = o3d.utility.Vector3dVector(rgb_frame_ref_cloud.transpose())
# vis.add_geometry(pcd)
# vis.add_geometry(origin)
# vis.update_renderer()
# print('x')
# while 1:
#     # vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()

rgb_frame_ref_cloud = np.asarray(pcd.points).transpose()
print('shape pf left_frame_ref_cloud')
print(rgb_frame_ref_cloud.shape)

rgb_frame_ref_cloud_normalized = rgb_frame_ref_cloud / rgb_frame_ref_cloud[2,:]
rgb_image_pts = np.matmul(M_RGB, rgb_frame_ref_cloud_normalized)

# Project this back to rgb image using cv2.projectPoints(objectPoin..) cannot use this now
# rgb_image_pts = cv2.projectPoints(rgb_frame_ref_cloud, cameraMatrix=M_RGB)
print('Transformed cloud shape')
print(rgb_frame_ref_cloud.shape)
print(rgb_frame_ref_cloud[:,index])

print('left_image_pts.shape')
print(rgb_image_pts.shape)
print(rgb_image_pts[:,index])

# print('Scaled RGB pts')
# scaled_rgb = rgb_image_pts / rgb_image_pts[2,:]
# print(scaled_rgb[:,index])
# print(scaled_rgb.reshape())
# depth_map_rgb = np

depth_rgb = np.full((720, 1280),  65535, dtype=np.uint16)
valid_pts_count = 0
# start = time.time()

for i in range(rgb_image_pts.shape[1]):
    x, y, w = rgb_image_pts[:, i]
    # pass
    x_int = int(round(x))
    y_int = int(round(y))
    if x_int >= 0 and x_int < 1280 and y_int >= 0 and y_int < 720:
        depth_rgb[y_int, x_int] =  int(rgb_frame_ref_cloud[2, i]*10) #converting back to mm
        valid_pts_count += 1

print('creating image')
cv2.imshow('egb_depth', depth_rgb)

# backtorgb = cv2.cvtColor(depth_vals,cv2.COLOR_GRAY2RGB)
# print(color.shape)
# print(backtorgb.shape)
# added_image = cv2.addWeighted(color,0.4,backtorgb,0.1,0)
# cv2.imshow('RGBD overlay ', added_image)
# cv2.waitKey()

depth_rgb[depth_rgb == 0] = 65535

im_color = (65535 // depth_rgb).astype(np.uint8)
                            # colorize depth map, comment out code below to obtain grayscale
im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_HOT)
cv2.imshow('color_map left', im_color)

# backtorgb = cv2.cvtColor(depth_rgb, cv2.COLOR_GRAY2RGB)
# print(color.shape)
# print(backtorgb.shape)
added_image = cv2.addWeighted(color,0.6,im_color,0.3,0)
cv2.imshow('RGBD overlay ', added_image)
cv2.waitKey()

# right_trasns = cv2.warpPerspective(right, F, frame.shape[::-1],
#                             cv2.INTER_CUBIC +
#                             cv2.WARP_FILL_OUTLIERS +
#                             cv2.WARP_INVERSE_MAP)

# right_trasns2 = cv2.warpPerspective(right, H_c, frame.shape[::-1],
#                             cv2.INTER_CUBIC +
#                             cv2.WARP_FILL_OUTLIERS +
#                             cv2.WARP_INVERSE_MAP)

            
# backtorgb = cv2.cvtColor(right_trasns2,cv2.COLOR_GRAY2RGB)
#             # backtorgb =  backtorgb[40:720 + 40, :]
# added_image = cv2.addWeighted(color,0.6,backtorgb,0.1,0)
# cv2.imshow('RGB-gray H-c overlay ', added_image)

            
            

# depth_vals = cv2.warpPerspective(depth_vals, H_c, depth_vals.shape[::-1],
#                             cv2.INTER_CUBIC +
#                             cv2.WARP_FILL_OUTLIERS +
#                             cv2.WARP_INVERSE_MAP)



# 1. change 1080 shape. 
# 2. crop the intrinisc matrix approprietly 
# 3. change depth in rectified right using homography to place it back in right frame and then rotate and translate it to rgb
# 4. how to handle this scenario when undistorted using mesh ? should I add distortions back ? 
# 5. What would be the best way to illuminate the lights properly to avoid reflections or bad calibration (Does vicalib overcomes this issue or is it universal for that too) 
# 6. Do we need calib to be in 4K ? I am thinking of doing it only for 1080 
# 7. Any suggestions on best way to handle in when using camera with auto focus ? 
# currently I have set it to a specific distance that helps in better focusing the calibration board with current setting
# we can create api to return the homography to place the depth from rectified right to rgb a.k.a center of the 1098OBC 
# or we can internally use wrap engine to do that before returning (extra load on Mx) 
# Cropping issue - center crop or bottom crop 
# ANother option is we can just find homography between right and rgb


