#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
import open3d as o3d
import time



# StereoDepth config options. TODO move to command line options
out_depth      = True  # Disparity by default
out_rectified  = False # Disparity by default# Output and display rectified streams
lrcheck  = False   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data

# right_intrinsic = [[855.000122,    0.000000,  644.814514],
#                     [0.000000,  855.263794,  407.305450],
#                     [0.000000,    0.000000,    1.000000]]

req_resolution = (720,1280)

# The following information is copied from the calibration print data that happens at the begining of gen1 depthai
# To have this printed on the terminal go to pcl-projection-rgb folder. Install the requrements and execute one of the examples there.
# P.S: This example doesn't work if you have recalibrated your device.

'''
 Rectification Rotation R2 (right): <- R2_right variable
    0.999763,   -0.012779,   -0.017639,
    0.012732,    0.999915,   -0.002802,
    0.017673,    0.002577,    0.999840,
  Calibration intrinsic matrix M1 (left):
  855.849548,    0.000000,  632.435974,
    0.000000,  856.289001,  399.700226,
    0.000000,    0.000000,    1.000000,
  Calibration intrinsic matrix M2 (right):    <- M_right variable
  855.000122,    0.000000,  644.814514,
    0.000000,  855.263794,  407.305450,
    0.000000,    0.000000,    1.000000,
  Calibration rotation matrix R:
    0.999903,    0.011196,    0.008257,
   -0.011240,    0.999922,    0.005380,
   -0.008196,   -0.005472,    0.999951,
  Calibration translation matrix T:
   -7.494308,
    0.095795,
    0.132222,
  Calibration Distortion Coeff d1 (Left):
   -4.481964,   14.138410,    0.002012,    0.000149,  -13.193083,   -4.541109,   14.358990,
  -13.394559,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
  Calibration Distortion Coeff d2 (Right):
   -5.598158,   19.114412,    0.000495,    0.000686,  -20.956785,   -5.645601,   19.298323,
  -21.132698,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
  Calibration intrinsic matrix M3 (rgb):  <- M_rgb variable dataset 
 1479.458984,    0.000000,  950.694458,
    0.000000, 1477.587158,  530.697632,
    0.000000,    0.000000,    1.000000,
  RGB-Right rotation matrix R (rgb):  <- R variable
    0.999986,    0.004985,    0.001887,
   -0.004995,    0.999974,    0.005245,
   -0.001861,   -0.005254,    0.999984,
 Calibration Distortion Coeff d3 (rgb):
   -1.872860,   16.683033,    0.001053,   -0.002063,   61.878521,   -2.158907,   18.424637,
   57.682858,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,    0.000000,
 Calibration translation vector T (rgb):  <- data in T_neg variable
   -3.782213,
    0.002144,
    0.122242,
'''

T_neg = -1 * np.array([   -3.782213, 0.002144, 0.122242])

R2_right = np.array([[0.999763,   -0.012779,   -0.017639],
                     [0.012732,    0.999915,   -0.002802],
                     [0.017673,    0.002577,    0.999840]])

R       = np.array([[0.999986,    0.004985,    0.001887],
              [-0.004995,    0.999974,    0.005245],
              [-0.001861,   -0.005254,    0.999984]])

M_right = np.array([[855.000122,    0.000000,  644.814514],
                    [0.000000,  855.263794,  407.305450],
                    [0.000000,    0.000000,    1.000000]]) 

# rgb is calibrated at 1080 p
M_RGB = np.array([[1479.458984,    0.000000,  950.694458],
                  [0.000000, 1477.587158,  530.697632],
                  [0.000000,    0.000000,    1.000000]])

M_right [1,2] -= 40


R_inv = np.linalg.inv(R)
H       = np.matmul(np.matmul(M_right, R2_right), np.linalg.inv(M_right))
H_inv   = np.linalg.inv(H)


scale_width = 1280/1920
m_scale = [[scale_width,      0,   0],
            [0,         scale_width,        0],
            [0,             0,         1]]


M_RGB = np.matmul(m_scale, M_RGB)
K_inv = np.linalg.inv(M_right)
inter_conv = np.matmul(K_inv, H_inv)


extrensics = np.hstack((R_inv, np.transpose([T_neg])))
transform_matrix = np.vstack((extrensics, np.array([0, 0, 0, 1])))


pcl_converter = None
# if point_cloud:
#     if out_rectified:
#         try:
#             from projector_3d import PointCloudVisualizer
#         except ImportError as e:
#             raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
#         pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
#     else:
#         print("Disabling point-cloud visualizer, as out_rectified is not set")

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_mono_cam_pipeline():
    print("Creating pipeline: MONO CAMS -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    cam        = pipeline.createColorCamera()
    xout_video = pipeline.createXLinkOut()

    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    xout_left .setStreamName('left')
    xout_right.setStreamName('right')

    cam_left .out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    streams = ['left', 'right']

    return pipeline, streams

def create_stereo_depth_pipeline():
    print("Creating Stereo Depth pipeline: ", end='')
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    cam_rgb    = pipeline.createColorCamera()
    stereo     = pipeline.createStereoDepth()
    xout_depth = pipeline.createXLinkOut()
    xout_video = pipeline.createXLinkOut()

    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)
    
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(True)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_video   .setStreamName('rgb_video')
    cam_rgb.video.link(xout_video.input)

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    
    stereo.setInputResolution(1280, 720)

    xout_depth   .setStreamName('depth')
    cam_left.out .link(stereo.left)
    cam_right.out.link(stereo.right)
    stereo.depth .link(xout_depth.input)

    return pipeline

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


# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    global last_rectif_right
    baseline = 75 #mm
    focal = M_right[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video': # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if pcl_converter is not None:
            if 0: # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
            else: # Option 2: project rectified right
                pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
            pcl_converter.visualize_pcd()
    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame


color = None
depth = None
pixel_coords = pixel_coord_np(1280, 720) 

def test_pipeline():
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline = create_stereo_depth_pipeline()

    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()
        # Create a receive queue for each stream
        q_list = []
        for s in ['rgb_video', 'depth']:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)

        # Need to set a timestamp for input frames, for the sync stage in Stereo node
        timestamp_ms = 0
        index = 0
        while True:
            # Handle input streams, if any
            for q in q_list:
                name  = q.getName()
                image = q.get()
                #print("Received frame:", name)
                # Skip some streams for now, to reduce CPU load
                frame = convert_to_cv2_frame(name, image)                
                
                if name == 'rgb_video':
                    scale_width = req_resolution[1]/frame.shape[1]

                    dest_res = (int(frame.shape[1] * scale_width), int(frame.shape[0] * scale_width)) ## opencv format dimensions
                    color = cv2.resize(
                        frame, dest_res, interpolation=cv2.INTER_CUBIC) # can change interpolation if needed to reduce computations
                if name == 'depth':
                    depth = frame.copy()

            if depth is None or color is None:
                continue


            temp = depth.copy() # depth in right frame
            cam_coords = np.dot(inter_conv, pixel_coords) * temp.flatten() * 0.1 # [x, y, z]
            del temp

            cam_coords[:, cam_coords[2] > 1500] = float('inf') 
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_coords.transpose())
            pcd.remove_non_finite_points()
            pcd.transform(transform_matrix)
            
            rgb_frame_ref_cloud = np.asarray(pcd.points).transpose()
            # print('shape pf left_frame_ref_cloud')
            # print(rgb_frame_ref_cloud.shape)
            rgb_frame_ref_cloud_normalized = rgb_frame_ref_cloud / rgb_frame_ref_cloud[2,:]
            rgb_image_pts = np.matmul(M_RGB, rgb_frame_ref_cloud_normalized)
            rgb_image_pts = rgb_image_pts.astype(np.int16)            
            # print("shape is {}".format(rgb_image_pts.shape[1]))            
            u_v_z = np.vstack((rgb_image_pts, rgb_frame_ref_cloud[2, :]))
        
            lft = np.logical_and(0 <= u_v_z[0], u_v_z[0] < 1280)
            rgt = np.logical_and(0 <= u_v_z[1], u_v_z[1] < 720)
            idx_bool = np.logical_and(lft, rgt)
            u_v_z_sampled = u_v_z[:, np.where(idx_bool)]
            y_idx = u_v_z_sampled[1].astype(int)
            x_idx = u_v_z_sampled[0].astype(int)

            depth_rgb = np.full((720, 1280),  65535, dtype=np.uint16)
            depth_rgb[y_idx,x_idx] = u_v_z_sampled[3]*10

            end = time.time()
            # print('for loop Convertion time')
            # print(end - start)
            cv2.imshow('rgb_depth', depth_rgb)
            
            depth_rgb[depth_rgb == 0] = 65535

            im_color = (65535 // depth_rgb).astype(np.uint8)
            # colorize depth map, comment out code below to obtain grayscale
            im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_HOT)
            
            added_image = cv2.addWeighted(color,0.6,im_color,0.3,0)
            cv2.imshow('RGBD overlay ', added_image)


                # cv2.imshow(name, frame)
            if cv2.waitKey(1) == ord('q'):
                break


test_pipeline()
