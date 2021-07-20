import cv2
import argparse
import numpy as np
import depthai as dai
import pygame
from pygame.locals import *
from pygame_checkbox import Checkbox, pygame_render_text
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = '1000,10'
pygame.init()

# color codings
white  = [255, 255, 255]
orange = [143, 122, 4]
red    = [230, 9, 9]
green  = [4, 143, 7]
black  = [0, 0, 0]

# parser = argparse.ArgumentParser()
# parser.add_argument("-ip", "--imgPath", type=str ,required=True,
#                             help="Path of the image file on which blur is calculated")
# args = parser.parse_args()

test_type = "Focus Test"
screen = pygame.display.set_mode((900, 700))

screen.fill(white)
pygame.display.set_caption(test_type)

title = "FOCUS TEST IN PROGRESS"
pygame_render_text(screen, title, (200,20), orange, 50)

if 'OAK_1' in test_type:
    auto_checkbox_names = ["RGB Focus"]
else:
    auto_checkbox_names = ["RGB Focus", "LEFT Focus", "RIGHT Focus"]

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
        rgb_cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb_cam.setInterleaved(False)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setFps(10)
        rgb_cam.setIspScale(1, 3)

        xout_rgb_isp = pipeline.createXLinkOut()
        xout_rgb_isp.setStreamName("rgb")
        rgb_cam.isp.link(xout_rgb_isp.input)        

    return pipeline


y = 110
x = 200

font = pygame.font.Font(None, 20)
auto_checkbox_dict = {}
for i in range(len(auto_checkbox_names)):
    w, h = font.size(auto_checkbox_names[i])
    x_axis = x - w
    y_axis = y +  (40*i)
    font_surf = font.render(auto_checkbox_names[i], True, green)
    screen.blit(font_surf, (x_axis,y_axis))
    auto_checkbox_dict[auto_checkbox_names[i]] = Checkbox(screen, x + 10, y_axis-5, outline_color=green, 
                                                check_color=green)
    auto_checkbox_dict[auto_checkbox_names[i]].render_checkbox()

# adding save button
# save_button =  pygame.Rect(600, 430, 60, 35)
# pygame.draw.rect(screen, orange, save_button)
# pygame_render_text(screen, 'SAVE', (605, 440))
pygame.display.update()

is_saved = False
focusThreshold = 20

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

max_count = 90

is_focused = False
is_finished = False

rgb_count   = 0
left_count  = 0
right_count = 0
lens_position = 0

while not is_focused:
    pygame.display.update()

    if enabledLR:
        left_frame = left_camera_queue.getAll()[-1]
        right_frame = right_camera_queue.getAll()[-1]
        
        dst_left = cv2.Laplacian(left_frame.getCvFrame(), cv2.CV_64F)
        abs_dst_left = cv2.convertScaleAbs(dst_left)
        mu, sigma_left = cv2.meanStdDev(dst_left)

        dst_right = cv2.Laplacian(right_frame.getCvFrame(), cv2.CV_64F)
        abs_dst_right = cv2.convertScaleAbs(dst_right)
        mu, sigma_right = cv2.meanStdDev(dst_right)
        print("SdtDev of Left: {} and right: {}".format(sigma_left, sigma_right))
        if sigma_right > focusThreshold:
            auto_checkbox_dict["RIGHT Focus"].check()
            right_count += 1
        else:
            if left_count > max_count:
                auto_checkbox_dict["RIGHT Focus"].uncheck()
            right_count += 1

        if sigma_left > focusThreshold:
            auto_checkbox_dict["LEFT Focus"].check()
            left_count += 1
        else:
            if right_count > max_count:
                auto_checkbox_dict["LEFT Focus"].uncheck()
            left_count += 1

        
    if enabledRGB:
        rgb_frame = rgb_camera_queue.getAll()[-1]
        recent_color = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2GRAY)
        print("In here")
        dst_rgb = cv2.Laplacian(recent_color, cv2.CV_64F)
        abs_dst_rgb = cv2.convertScaleAbs(dst_rgb)
        mu, sigma_rgb = cv2.meanStdDev(dst_rgb)
        print("SdtDev of RGB: {} with lens position of {}".format(sigma_rgb, rgb_frame.getLensPosition()))
        # cv2.imshow(" Blurred Image", recent_color)
        # cv2.waitKey(1)

        if sigma_rgb > focusThreshold:
            lens_position = rgb_frame.getLensPosition()
            auto_checkbox_dict["RGB Focus"].check()
            rgb_count += 1
        else:
            if rgb_count > max_count:
                auto_checkbox_dict["RGB Focus"].uncheck()
            rgb_count += 1

    for checks in auto_checkbox_dict.values():
        checks.render_checkbox()

    if enabledLR and enabledRGB:
        print("right count: {}, left count: {} and rgb count: {}".format(right_count, left_count, rgb_count))
        if right_count > max_count and left_count > max_count and rgb_count > max_count:
            all_checked = True
            for checks in auto_checkbox_dict.values():
                all_checked = all_checked and checks.is_checked()
            is_focused = all_checked
    elif enabledRGB:
        if rgb_count > max_count:
            all_checked = True
            for checks in auto_checkbox_dict.values():
                all_checked = all_checked and checks.is_checked()
            is_focused = all_checked
    else:
        raise RuntimeError("Invalid Option")


all_checked = True
for checks in auto_checkbox_dict.values():
    all_checked = all_checked and checks.is_checked()

print("Current Lens Position is {}".format(lens_position))
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
