#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import cv2
import depthai
from projector_3d import PointCloudVisualizer
import numpy as np


def cvt_to_bgr(packet):
    meta = packet.getMetadata()
    w = meta.getFrameWidth()
    h = meta.getFrameHeight()
    # print((h, w))
    packetData = packet.getData()
    yuv420p = packetData.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

curr_dir = str(Path('.').resolve().absolute())

device = depthai.Device("", False)
pipeline = device.create_pipeline(config={
    'streams': ['right', 'depth', 'color'],
    'ai': {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
    },
    'camera': {'mono': {'resolution_h': 720, 'fps': 30},
                'rgb':{'resolution_h': 2160, 'fps': 30}},
})

if pipeline is None:
    raise RuntimeError("Error creating a pipeline!")

right = None
pcl_converter = None
color = None
# req resolution in numpy format
req_resolution = (720,1280) # (h,w) -> numpy format. opencv format (w,h)

while True:
    data_packets = pipeline.get_available_data_packets()

    for packet in data_packets:
        if packet.stream_name == "color":
            color = cvt_to_bgr(packet)
            print(color.shape) # numpy format (h, w)
            final_path = curr_dir + '/dataset/2160.png'
            print(final_path)
            cv2.imwrite(final_path, color)
            scale_width = req_resolution[1]/color.shape[1]
            dest_res = (int(color.shape[1] * scale_width), int(color.shape[0] * scale_width)) ## opencv format dimensions
            # print("destination resolution------>")
            # print(dest_res)
            color = cv2.resize(
                color, dest_res, interpolation=cv2.INTER_CUBIC) # can change interpolation if needed to reduce computations
            # print("scaled gray shape")
            # print(gray.shape)
            
            if color.shape[0] < req_resolution[0]: # height of color < required height of image
                raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                    color.shape[0], req_resolution[0]))
            # print(gray.shape[0] - req_resolution[0])
            del_height = (color.shape[0] - req_resolution[0]) // 2
            ## TODO(sachin): change center crop and use 1080 directly and test
            color = color[del_height: del_height + req_resolution[0], :]
            # final_path = curr_dir + '/dataset/resized_scaled.png'
            # print(final_path)
            # cv2.imwrite(final_path, color)
            print("scaled color frame shape")
            print(color.shape)
            cv2.imshow('color resized', color)
        if packet.stream_name == "right":
            right = packet.getData()
            print(right.shape)
            
            # final_path = curr_dir + '/dataset/right.png'
            # print(final_path)
            # cv2.imwrite(final_path, right)
            cv2.imshow(packet.stream_name, right)
            
        elif packet.stream_name == "depth":
            frame = packet.getData()
            M2 = device.get_right_intrinsic()
            M3 = np.array([[2968.3318, 0, 2096.0703],
                            [0, 2968.3318, 1444.4983],
                            [0,     0,          1   ]], dtype=np.float32)
            # scaling rgb intrinsics from 4k to 720p
            scale_width = 1280/4056
            m_scale = np.array([[scale_width,      0,   0],
                    [0, scale_width,   0],
                    [0,      0,    1]], dtype=np.float32)
            M_RGB = np.matmul(m_scale, M3)
            height = round(3040 * scale_width)
            if height > 720:
                diff = (height - 720) // 2
                M_RGB[1, 2] -= diff
            M_RGB = np.array([[937.1739, 0, 659.10703],
                            [0, 935.9882, 378.6677],
                            [0,     0,          1   ]], dtype=np.float32)
            R = np.array([[0.999946, -0.008328, -0.006217],
                          [0.008445,  0.999782,  0.019107],
                          [0.006056, -0.019158,  0.999798]], dtype=np.float32)

            T = np.array([-3.746683, 0.012276, 0.319712], np.float32)
            R[:,2] =  T
            
            # H_forward = np.matmul(np.matmul(M_RGB, R), np.linalg.inv(M2))
            H_forward = (np.matmul(M_RGB, R))
            # H_inv = np.linalg.inv(device.get_right_homography())
            H_inv = np.matmul(H_forward, np.matmul(np.linalg.inv(M2),np.linalg.inv(device.get_right_homography())))


            H = np.linalg.inv(np.matmul(H_forward, H_inv))
            print("M_RGB")
            print(M_RGB)
            print("R----->")
            print(R)
            # np.linalg.inv
            H_forward = (np.matmul(np.matmul(M2, R), np.linalg.inv(M2)))
            depth_vals = cv2.warpPerspective(frame, H_inv, frame.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)
            cv2.imshow('color homo', depth_vals)
            if right is not None:
                if pcl_converter is None:
                    fd, path = tempfile.mkstemp(suffix='.json')
                    with os.fdopen(fd, 'w') as tmp:
                        json.dump({
                            "width": 1280,
                            "height": 720,
                            "intrinsic_matrix": [item for row in device.get_right_intrinsic() for item in row]
                        }, tmp)
                    # with os.fdopen(fd, 'w') as tmp:
                    #     json.dump({
                    #         "width": 1280,
                    #         "height": 720,
                    #         "intrinsic_matrix": [item.astype(float) for row in M_RGB for item in row]
                    #     }, tmp)
                    pcl_converter = PointCloudVisualizer(path)
                pcd = pcl_converter.rgbd_to_projection(frame, color)
                pcl_converter.visualize_pcd()
            cv2.imshow(packet.stream_name, frame)
    if cv2.waitKey(1) == ord("q"):
        break


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

if pcl_converter is not None:
    pcl_converter.close_window()
