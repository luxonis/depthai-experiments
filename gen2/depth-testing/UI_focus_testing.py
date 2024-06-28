import cv2
import argparse
import numpy as np
import depthai as dai
import os
import time

# color codings
red = (255, 0, 0)
green = (0, 255, 0)

parser = argparse.ArgumentParser()
parser.add_argument("-tm", "--testMode", type=str ,required=True,
                            help="Define the type of test. ex: oak_d")
args = parser.parse_args()
test_type = args.testMode

font = cv2.FONT_HERSHEY_SIMPLEX
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def create_pipeline(enableLR, enableRgb):
    pipeline = dai.Pipeline()

    if enableLR:
        cam_left = pipeline.createMonoCamera()
        cam_right = pipeline.createMonoCamera()

        xout_left = pipeline.createXLinkOut()
        xout_right = pipeline.createXLinkOut()

        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                
        cam_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)

        cam_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)
        cam_left.setFps(10)
        cam_right.setFps(10)

        xout_left.setStreamName("left")
        cam_left.out.link(xout_left.input)

        xout_right.setStreamName("right")
        cam_right.out.link(xout_right.input)

    if enableRgb:
        rgb_cam = pipeline.createColorCamera()
        controlIn = pipeline.createXLinkIn()

        rgb_cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb_cam.setInterleaved(False)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setFps(10)
        rgb_cam.setIspScale(1, 3)
        rgb_cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)

        controlIn.setStreamName('control')
        controlIn.out.link(rgb_cam.inputControl)

        xout_rgb_isp = pipeline.createXLinkOut()
        xout_rgb_isp.setStreamName("rgb")
        rgb_cam.isp.link(xout_rgb_isp.input)        

    return pipeline


enabledLR = False
enabledRGB = False

focusThreshold = 20
if 'oak_1' in test_type:
    enabledRGB = True
else:
    enabledLR = True
    enabledRGB = True

# TODO(sachin): Add detection and acceptance of board in LR before continuing  focal check 
device = dai.Device() 
pipeline = create_pipeline(enabledLR, enabledRGB)
device.startPipeline(pipeline)

if enabledLR:
    left_camera_queue = device.getOutputQueue("left", 5, False)
    right_camera_queue = device.getOutputQueue("right", 5, False)
if enabledRGB:
    rgb_camera_queue = device.getOutputQueue("rgb", 5, False)
    rgb_control_queue = device.getInputQueue("control", 5, False)

max_count   = 90
rgb_count   = 0
left_count  = 0
right_count = 0

lens_position = 0

is_left_focused = False
is_right_focused = False
is_rgb_focused = False
ctrl = dai.CameraControl()
ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
ctrl.setAutoFocusTrigger()

rgb_control_queue.send(ctrl)

trig_count = 0
while True:
    if enabledLR:
        left_frame = left_camera_queue.getAll()[-1]
        right_frame = right_camera_queue.getAll()[-1]
        
        dst_left = cv2.Laplacian(left_frame.getCvFrame(), cv2.CV_64F)
        # abs_dst_left = cv2.convertScaleAbs(dst_left)
        mu, sigma_left = cv2.meanStdDev(dst_left)

        dst_right = cv2.Laplacian(right_frame.getCvFrame(), cv2.CV_64F)
        # abs_dst_right = cv2.convertScaleAbs(dst_right)
        mu, sigma_right = cv2.meanStdDev(dst_right)
        print("SdtDev of Left: {} and right: {}".format(sigma_left, sigma_right))

        if sigma_right > focusThreshold:
            is_right_focused = True
        right_count += 1

        if sigma_left > focusThreshold:
            is_left_focused = True
        left_count += 1

    if enabledRGB:
        rgb_frame = rgb_camera_queue.getAll()[-1]
        recent_color = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2GRAY)

        marker_corners, _, _ = cv2.aruco.detectMarkers(recent_color, aruco_dictionary)
        if len(marker_corners) < 20:
            print("Board not detected. Waiting...!!!")
            trig_count += 1
            rgb_count += 1
            if trig_count > 31:
                trig_count = 0
                rgb_control_queue.send(ctrl)
                time.sleep(1)
            if rgb_count > max_count:
                break
            continue

        dst_rgb = cv2.Laplacian(recent_color, cv2.CV_64F)
        # abs_dst_rgb = cv2.convertScaleAbs(dst_rgb)
        mu, sigma_rgb = cv2.meanStdDev(dst_rgb)
        print("SdtDev of RGB: {} with lens position of {}".format(sigma_rgb, rgb_frame.getLensPosition()))
        cv2.imshow(" Recent Color Image", recent_color)
        cv2.waitKey(1)

        if sigma_rgb > focusThreshold:
            lens_position = rgb_frame.getLensPosition()
            is_rgb_focused = True
        else:
            trig_count += 1
            if trig_count > 31:
                trig_count = 0
                rgb_control_queue.send(ctrl)
                time.sleep(1)
            
        rgb_count += 1

    if enabledLR and enabledRGB:
        print("right count: {}, left count: {} and rgb count: {}".format(right_count, left_count, rgb_count))
        if right_count > max_count and left_count > max_count and rgb_count > max_count:
            break
    elif enabledRGB:
        if rgb_count > max_count:
            break
    else:
        raise RuntimeError("Invalid Option")


if is_rgb_focused:
    calibration_handler = dai.CalibrationHandler()
    calibration_handler.setLensPosition(dai.CameraBoardSocket.RGB, lens_position)
    device.flashCalibration(calibration_handler)
    print("Calibration Flashed")

image = create_blank(512, 512, rgb_color=red)
if enabledLR and enabledRGB:
    if is_rgb_focused and is_left_focused and is_right_focused:
        image = create_blank(512, 512, rgb_color=green)
        cv2.putText(image,'Focus TEST',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'PASSED',(10,300), font, 2,(0,0,0),2)
    elif not is_rgb_focused:
        image = create_blank(512, 512, rgb_color=red)
        cv2.putText(image,'RGB Focus TEST ',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)
    elif not is_right_focused:
        image = create_blank(512, 512, rgb_color=red)
        cv2.putText(image,'Right Focus TEST ',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)
    elif not is_left_focused:
        image = create_blank(512, 512, rgb_color=red)
        cv2.putText(image,'Left Focus TEST ',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)

elif enabledRGB:
    if is_rgb_focused:
        image = create_blank(512, 512, rgb_color=green)
        cv2.putText(image,'Focus TEST',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'PASSED',(10,300), font, 2,(0,0,0),2)
    else:
        image = create_blank(512, 512, rgb_color=red)
        cv2.putText(image,'RGB Focus TEST ',(10,250), font, 2,(0,0,0),2)
        cv2.putText(image,'FAILED',(10,300), font, 2,(0,0,0),2)

print("Current Lens Position is {}".format(lens_position))
cv2.imshow("Result Image", image)
cv2.waitKey(0)

""" src_gray = cv2.imread(args.imgPath, cv2.IMREAD_GRAYSCALE)

dst = cv2.Laplacian(src_gray, cv2.CV_64F)

abs_dst = cv2.convertScaleAbs(dst)

# cv2.Scalar mu, sigma;
mu, sigma = cv2.meanStdDev(dst)

print("Image Mean: {} and StdDev: {}".format(mu, sigma))

cv2.imshow(" Blurred Image", blur)
cv2.imshow("Laplace Image", abs_dst)
cv2.waitKey(0)
 """
