#!/usr/bin/env python3

import cv2
import numpy as np

def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def alignDepthToRGB(depth_image, pixel_coords, inverse_depth_intrinsic, rgb_intrinsic, depth_to_rgb_extrinsic, rgb_width, rgb_height):
    """
    Align the depth image with the color image. 
    Returns:
        Aligned depth image
    """
    depth_scale = 10.0

    # depth to 3d coordinates [x, y, z]
    cam_coords = np.dot(inverse_depth_intrinsic, pixel_coords) * \
        depth_image.flatten().astype(float) / depth_scale

    # move the depth image 3d coordinates to the rgb camera  location
    cam_coords_homogeneous = np.vstack(
        (cam_coords, np.ones((1, cam_coords.shape[1]))))
    depth_points_homogeneous = np.dot(
        depth_to_rgb_extrinsic, cam_coords_homogeneous)

    # project the 3d depth points onto the rgb image plane
    rgb_frame_ref_cloud = depth_points_homogeneous[:3, :]
    rgb_frame_ref_cloud_normalized = rgb_frame_ref_cloud / \
        rgb_frame_ref_cloud[2, :]
    rgb_image_pts = np.matmul(rgb_intrinsic, rgb_frame_ref_cloud_normalized)
    rgb_image_pts = rgb_image_pts.astype(np.int16)
    u_v_z = np.vstack((rgb_image_pts, rgb_frame_ref_cloud[2, :]))
    lft = np.logical_and(0 <= u_v_z[0], u_v_z[0] < rgb_width)
    rgt = np.logical_and(0 <= u_v_z[1], u_v_z[1] < rgb_height)
    idx_bool = np.logical_and(lft, rgt)
    u_v_z_sampled = u_v_z[:, np.where(idx_bool)]
    y_idx = u_v_z_sampled[1].astype(int)
    x_idx = u_v_z_sampled[0].astype(int)

    # place the valid aligned points into a new depth image
    aligned_depth_image = np.full((rgb_height, rgb_width), 0, dtype=np.uint16)
    aligned_depth_image[y_idx, x_idx] = u_v_z_sampled[3]*depth_scale
    return aligned_depth_image


def quantizeDepthFrame(frame, depth_scale_factor):
    """"
    Further quantize the depth image for nice visualization
    depth_scale_factor == value used to normalize depth image
    Returns:
        Quantized depth map image
    """
    quantized_depth = cv2.applyColorMap(cv2.convertScaleAbs(frame.astype(float), alpha=255/depth_scale_factor),cv2.COLORMAP_JET)
    return quantized_depth

def overlayDepthFrame(rgb_frame, depth_frame, rgb_alpha):
    """"
    Overlay depth image onto rgb image.
    Returns:
        RGB image with depth map overlay
    """
    depth_three_channel = np.zeros_like(rgb_frame)
    depth_three_channel[:, :, 2] = depth_frame
    cond = depth_three_channel[:, :, 2] > 0
    depth_three_channel[cond, 2] = 255
    # Blend aligned depth + rgb image
    blended_image = (1.0 - rgb_alpha) * depth_three_channel.astype(float) + rgb_alpha * rgb_frame.astype(float)
    blended_image = (255 * blended_image.astype(float) / blended_image.max()).astype(np.uint8)
    return blended_image
