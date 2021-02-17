from stereo_on_host import StereoSGBM
import cv2
import numpy as np
import glob
from pathlib import Path
import depthai

def cvt_to_bgr(packet):
    meta = packet.getMetadata()
    w = meta.getFrameWidth()
    h = meta.getFrameHeight()
    # print((h, w))
    packetData = packet.getData()
    yuv420p = packetData.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

device = depthai.Device("", False)

pipeline = device.create_pipeline(config={
    'streams': ['right', 'color', 'left'],
    'ai': {
        "blob_file": str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
    },
    'camera': {'mono': {'resolution_h': 720, 'fps': 30},
                'rgb':{'resolution_h': 1080, 'fps': 30}},
})

cam_c = depthai.CameraControl.CamId.RGB
device.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO)
cmd_set_focus = depthai.CameraControl.Command.MOVE_LENS
device.send_camera_control(cam_c, cmd_set_focus, '135')

M_rgb   = np.array(device.get_intrinsic(depthai.CameraControl.CamId.RGB))
d_rgb   = np.array(device.get_distortion_coeffs(depthai.CameraControl.CamId.RGB))

M_left  = np.array(device.get_left_intrinsic())
d_left  = np.array(device.get_distortion_coeffs(depthai.CameraControl.CamId.LEFT))

R_l_r   = np.array(device.get_rotation())
T_l_r   = np.array(device.get_translation())

R_r_rgb = np.array(device.get_rgb_rotation())
T_r_rgb = np.array(device.get_rgb_translation())

M_right = np.array(device.get_right_intrinsic())
d_right  = np.array(device.get_distortion_coeffs(depthai.CameraControl.CamId.RIGHT))

scale_width = 1280/1920
m_scale = [[scale_width,      0,   0],
            [0,         scale_width,        0],
            [0,             0,         1]]

print(M_rgb.shape)
M_rgb = np.matmul(m_scale, M_rgb)

# Project_r_rgb = np.hstack((R_r_rgb, T_r_rgb))
# Project_l_r = np.hstack((R_r_rgb, T_r_rgb))

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M_right, d_right, M_rgb, d_rgb,(1280,720), R_r_rgb, T_r_rgb)

H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_rgb))
H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))           

stereo_obj = StereoSGBM(3.25, H_right, H_left)

right = None
gray_rgb = None

while True:
    data_packets = pipeline.get_available_data_packets(True)
    
    for packet in data_packets:
        if packet.stream_name == "color":
            color = cvt_to_bgr(packet)
            gray_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            scale_width = 1280 / color.shape[1]
            dest_res = (int(color.shape[1] * scale_width), int(color.shape[0] * scale_width))
            gray_rgb = cv2.resize(gray_rgb, dest_res, interpolation=cv2.INTER_CUBIC)
        if packet.stream_name =='right':
            right = packet.getData()
            # print(left.shape[::-1])
            # left, right = stereo_obj.rectification(gray_rgb, right)
            if right is not None and gray_rgb is not None:
                stereo_obj.create_disparity_map(gray_rgb, right)
                



# if __name__ == "__main__":

#     left_images_paths = glob.glob("stereo_on_host/dataset/left" + '/left*.png')
#     right_images_paths = glob.glob("stereo_on_host/dataset/right" + '/right*.png')
#     H_right = np.array([[1.012611746788025, -0.009315459989011288, -16.0528621673584],
#                         [0.017922570928931236, 1.0018900632858276, -16.368852615356445], 
#                         [1.9619521481217816e-05, 2.267290710733505e-06, 0.9867464303970337]])

#     # H_left = np.array([[1.0085175037384033, 0.02535983733832836, -24.814842224121094],
#     #                     [-0.021759534254670143, 0.998845636844635, 15.92939281463623],
#     #                     [1.4236070455808658e-05, -2.179780722144642e-06, 0.9917676448822021]])
#     left_images_paths.sort()
#     right_images_paths.sort()
#     print(left_images_paths)
#     stereo_process = StereoSGBM(7.5, H_right) # baselienght and right homography. 

#     for left_img_path, right_img_path in zip(left_images_paths, right_images_paths):
#         stereo_process.create_disparity_map(left_img_path, right_img_path)
