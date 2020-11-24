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
    'camera': {'mono': {'resolution_h': 800, 'fps': 30},
                'rgb':{'resolution_h': 1080, 'fps': 30}},
})

cam_c = depthai.CameraControl.CamId.RGB
device.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO)
cmd_set_focus = depthai.CameraControl.Command.MOVE_LENS
device.send_camera_control(cam_c, cmd_set_focus, '111')

# sleep(2)

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
            # print(color.shape) # numpy format (h, w)
            # final_path = curr_dir + '/dataset/3040.png'
            # print(final_path)
            # cv2.imwrite(final_path, color)
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
            color_center = color[del_height: del_height + req_resolution[0], :]
            # final_path = curr_dir + '/dataset/resized_scaled_center.png'
            # print(final_path)
            # cv2.imwrite(final_path, color_center)
            # print("scaled color frame shape")
            # print(color_center.shape)
            cv2.imshow('color resized', color_center)

            # color_top = color[: req_resolution[0], :]
            # final_path = curr_dir + '/dataset/resized_scaled_top.png'
            # print(final_path)
            # cv2.imwrite(final_path, color_top)
            
            # color_bottom = color[del_height*2: req_resolution[0], :]
            # final_path = curr_dir + '/dataset/resized_scaled_bottom.png'
            # print(final_path)
            # cv2.imwrite(final_path, color_bottom)
            color = color_center

        if packet.stream_name == "right":
            right = packet.getData()
            # print(right.shape)
            
            # final_path = curr_dir + '/dataset/right.png'
            # print(final_path)
            # cv2.imwrite(final_path, right)
            cv2.imshow(packet.stream_name, right)
            
        elif packet.stream_name == "depth":
            frame = packet.getData()
            M2 = device.get_right_intrinsic()
            print("Displaying M2")
            print(M2)
            print()
            # M3 = np.array([[2968.3318, 0, 2096.0703],
            #                 [0, 2968.3318, 1444.4983],
            #                 [0,     0,          1   ]], dtype=np.float32)
            # # scaling rgb intrinsics from 4k to 720p
            # scale_width = 1280/4056
            # m_scale = np.array([[scale_width,      0,   0],
            #         [0, scale_width,   0],
            #         [0,      0,    1]], dtype=np.float32)
            # M_RGB = np.matmul(m_scale, M3)
            # height = round(3040 * scale_width)
            # if height > 720:
            #     diff = (height - 720) // 2
            #     M_RGB[1, 2] -= diff
            # M_RGB = np.array([[937.1739, 0, 659.10703],
            #                 [0, 935.9882, 378.6677],
            #                 [0,     0,          1   ]], dtype=np.float32)
            M_RGB = np.array([[938.1863, 0, 659.42112274],
                            [0, 936.9993429, 458.2857],
                            [0,     0,          1   ]], dtype=np.float32)
            M_RGB = np.array([[986.76263743, 0, 637.06145685],
                              [0, 985.51417088, 356.9171903],
                              [0,     0,          1   ]], dtype=np.float32)
            R = np.array([[0.999946, -0.008328, -0.006446],
                          [0.008442,  0.999802,  0.017998],
                          [0.006295, -0.018051,  0.999817]], dtype=np.float32)


            print("R before")
            print(R)
            T = np.array([-3.746683, 0.012276, 0.319712], np.float32)
            # R[:,2] =  T
            print("R for homo")
            print(R)
            # H_forward = np.matmul(np.matmul(M_RGB, R), np.linalg.inv(M2))
            # H_forward = (np.matmul(M_RGB, R))
            H_inv = np.linalg.inv(device.get_right_homography())
            # H_inv = np.matmul(H_forward, np.matmul(np.linalg.inv(M2),np.linalg.inv(device.get_right_homography())))

            H_forward = np.matmul(M_RGB,np.matmul( R, np.linalg.inv(M2)))
            


            # H = np.linalg.inv(np.matmul(H_forward, H_inv))
            # print("M_RGB")
            # print(M_RGB)
            # print("R----->")
            # print(R)
            # np.linalg.inv
            # H_forward = (np.matmul(np.matmul(M2, R), np.linalg.inv(M2)))

            # using corners from all the images
            H1 = np.array([[1.08319806 , 7.129378e-02, 8.36212138],
                          [-1.15394512e-02,  1.091549,  -5.7350279e+01],
                          [-5.83017074e-06, 1.17000754e-05,  1]], dtype=np.float32)

            # using corners from first image
            H2 = np.array([[1.04303565,     3.851567e-03, -0.006217],
                          [-6.87673277e-03,  0.999782,  0.019107],
                          [5.79007130e-06,  -0.019158,  0.999798]], dtype=np.float32)

            # using corners from second image
            H3 = np.array([[0.999946, -0.008328, -0.006217],
                          [0.008445,  0.999782,  0.019107],
                          [0.006056, -0.019158,  0.999798]], dtype=np.float32)


            # using all images but rgb was top cropper while rest was center cropped
            # H_c = np.array([[1.08690918 , 7.41758715e-02, 6.7014219],
            #               [-1.13036787e-02,  1.09798085,  2.008645e+01],
            #               [-4.56661176e-06, 1.57753751e-05,  1]], dtype=np.float32)

            H_c = np.array([[1.134155 , 2.26799882e-02, -1.68295944e+01],
                          [-2.66053562e-02,  1.13454926,  -7.51904639e+01],
                          [-4.75453420e-06, 1.03164808e-05,  1]], dtype=np.float32)

            F = np.array([[-2.54356430e-09 , 6.24459718e-08, -2.69654978e-04],
                          [-1.22788882e-07,  -7.93706750e-09,  -1.50105710e-02],
                          [-6.28306896e-05, 1.3159833e-02,  1]], dtype=np.float32)

            # calculate Fundamental matrix Fundamental
            # Fundamental matrix doesnt workdir
            # Soln: add autofocus to have const in calibration code. 
            # Followed bty test H_c again using that and 
            # then using depth map convert it into 3D pts and then place them back into rgb intrinsics
            [-3.76482, 0.062066, 0.015372]
            s_tb =np.array([[0 ,        -0.015372,      0.062066],
                            [0.015372,  0,  -1.50105710e-02],
                            [-6.28306896e-05, 1.3159833e-02,  0]], dtype=np.float32)
  
            print("Expected homography")
            print(H_c)

            print("Calculated homography")
            print(H_forward)
            print("R form H_c homography")
            print(np.matmul(np.matmul(np.linalg.inv(M_RGB), H_c) ,M2))

            right_trasns = cv2.warpPerspective(right, F, frame.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            right_trasns2 = cv2.warpPerspective(right, H_c, frame.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            cv2.imshow('Right fundamental matrix', right_trasns)
            cv2.imshow('Right H_c', right_trasns2)
            
            backtorgb = cv2.cvtColor(right_trasns2,cv2.COLOR_GRAY2RGB)
            backtorgb =  backtorgb[40:720 + 40, :]
            added_image = cv2.addWeighted(color,0.6,backtorgb,0.1,0)
            cv2.imshow('RGB-gray H-c overlay ', added_image)

            depth_vals = cv2.warpPerspective(frame, H_inv, frame.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)
            

            depth_vals = cv2.warpPerspective(depth_vals, H_c, depth_vals.shape[::-1],
                                        cv2.INTER_CUBIC +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            cv2.imshow('color homo', depth_vals)
            frame = depth_vals

            # backtorgb = cv2.cvtColor(depth_vals,cv2.COLOR_GRAY2RGB)
            # print(color.shape)
            # print(backtorgb.shape)
            # added_image = cv2.addWeighted(color,0.4,backtorgb,0.1,0)
            # cv2.imshow('RGBD overlay ', added_image)

            if right is not None:
                if pcl_converter is None:
                    fd, path = tempfile.mkstemp(suffix='.json')
                    # with os.fdopen(fd, 'w') as tmp:
                    #     json.dump({
                    #         "width": 1280,
                    #         "height": 720,
                    #         "intrinsic_matrix": [item for row in device.get_right_intrinsic() for item in row]
                    #     }, tmp)
                    with os.fdopen(fd, 'w') as tmp:
                        json.dump({
                            "width": 1280,
                            "height": 720,
                            "intrinsic_matrix": [item.astype(float) for row in M_RGB for item in row]
                        }, tmp)
                #     pcl_converter = PointCloudVisualizer(path)
                # pcd = pcl_converter.rgbd_to_projection(frame, color)
                # pcl_converter.visualize_pcd()
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
