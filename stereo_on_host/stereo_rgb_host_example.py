from stereo_on_host import StereoSGBM
import cv2
import numpy as np
import glob
from pathlib import Path
import depthai as dai

def cvt_to_bgr(packet):
    meta = packet.getMetadata()
    w = meta.getFrameWidth()
    h = meta.getFrameHeight()
    # print((h, w))
    packetData = packet.getData()
    yuv420p = packetData.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

def create_pipeline():
    pipeline = dai.Pipeline()

    rgb_cam  = pipeline.createColorCamera()
    cam_left = pipeline.createMonoCamera()
    xout_left     = pipeline.createXLinkOut()
    xout_rgb_isp  = pipeline.createXLinkOut()

    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    rgb_cam.setInterleaved(False)
    rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.initialControl.setManualFocus(135)
    rgb_cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    rgb_cam.setIspScale(1, 3)

    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam_left.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

    xout_left.setStreamName("left")
    cam_left.out.link(xout_left.input)
    xout_rgb_isp.setStreamName("rgb")
    rgb_cam.isp.link(xout_rgb_isp.input)
    # rgb_cam.isp.link(xout_rgb_isp.input)
    return pipeline


pipeline = create_pipeline()
device = dai.Device(pipeline)
device.startPipeline()

calibObj = device.getCalibration()

# TODO (sachin) :rgb calibration returned is wrong when 720 res arg is passsed.
M_rgb   = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.RGB))
# d_rgb   = np.array(device.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
print("M rgb")
print(M_rgb)
scale_width = 1280/1920
m_scale = [[scale_width,      0,   0],
            [0,         scale_width,        0],
            [0,             0,         1]]
M_rgb = np.matmul(m_scale, M_rgb)
print("scaled M rgb")
print(M_rgb)
M_left  = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))

print("M M_left")
print(M_left)

# d_left  = np.array(calibObj.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))


# d_right  = np.array(device.get_distortion_coeffs(depthai.CameraControl.CamId.RIGHT))


# Project_r_rgb = np.hstack((R_r_rgb, T_r_rgb))
# Project_l_r = np.hstack((R_r_rgb, T_r_rgb))
R1 = np.array(calibObj.getStereoLeftRectificationRotation())
R2 = np.array(calibObj.getStereoRightRectificationRotation())
# R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M_right, d_right, M_rgb, d_rgb,(1280,720), R_r_rgb, T_r_rgb)

H_left = np.matmul(np.matmul(M_rgb, R1), np.linalg.inv(M_left))
H_rgb = np.matmul(np.matmul(M_rgb, R2), np.linalg.inv(M_rgb))           

print("H left")
print(H_left)

print("H rgb")
print(H_rgb)

print("H left inv")
print(np.linalg.inv(H_left))

print("H rgb inv")
print(np.linalg.inv(H_rgb))


stereo_obj = StereoSGBM(3.75, H_rgb, H_left)

left = None
gray_rgb = None

left_camera_queue = device.getOutputQueue("left", 5, False)
rgb_camera_queue  = device.getOutputQueue("rgb", 5, False)
while True:
    
    left_frame = left_camera_queue.tryGet()
    if left_frame is not None:
        left = left_frame.getCvFrame()
    rgb_frame = rgb_camera_queue.tryGet()
    if rgb_frame is not None:
        gray_rgb = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2GRAY)
        scale_width = 1280 / gray_rgb.shape[1]
        dest_res = (int(gray_rgb.shape[1] * scale_width), int(gray_rgb.shape[0] * scale_width))
        gray_rgb = cv2.resize(gray_rgb, dest_res, interpolation=cv2.INTER_CUBIC)
    if left is not None and gray_rgb is not None:
        stereo_obj.create_disparity_map(left, gray_rgb)
                