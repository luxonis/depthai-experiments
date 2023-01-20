import cv2
import argparse
import numpy as np
import depthai
import pygame
from pygame.locals import *

os.environ['SDL_VIDEO_WINDOW_POS'] = '1000,10'
pygame.init()

parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--imgPath", type=str ,required=True,
                            help="Path of the image file on which blur is calculated")
args = parser.parse_args()

src_gray = cv2.imread(args.imgPath, cv2.IMREAD_GRAYSCALE)
blur = cv2.blur(src_gray,(5,5))

dst = cv2.Laplacian(src_gray, cv2.CV_64F)
dst_blr = cv2.Laplacian(blur, cv2.CV_64F)

abs_dst = cv2.convertScaleAbs(dst)
abs_dst_blr = cv2.convertScaleAbs(dst_blr)

# cv2.Scalar mu, sigma;
mu, sigma = cv2.meanStdDev(dst)
mu_b, sigma_b = cv2.meanStdDev(dst_blr)

print("Image Mean: {} and StdDev: {}".format(mu, sigma))
print("Image blurred Mean: {} and StdDev: {}".format(mu_b, sigma_b))

cv2.imshow(" Blurred Image", blur)
cv2.imshow("Laplace Image", abs_dst)
cv2.imshow("Laplace Blurred Image", abs_dst_blr)

cv2.waitKey(0)