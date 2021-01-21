import cv2
import numpy as np
import os
import glob
import depthai





class StereoSGBM:
    def __init__(self, baseline, H_right, H_left = np.identity(3, dtype=np.float32)):
        self.max_disparity = 96
        self.stereoProcessor = cv2.StereoSGBM_create(0, self.max_disparity, 21)
        self.baseline = baseline
        fov = 71.86
        self.focal_lenght =  1280 /(2 * np.tan((fov * 3.142) / (2 * 180)))   # orig_frame_w / (2.f * std::tan(fov / 2 / 180.f * pi));
        print(self.focal_lenght)
        self.H1 = H_left  # for left camera
        self.H2 = H_right # for right camera

        # self.lut = []
        # for i in range(256): # Not in use yet.
        #     if i == 0:
        #         self.lut.append(65535)
        #         continue
        #     z_m = (self.focal_lenght * self.baseline) / i;
        #     self.lut.append(min(65535.0, z_m * 1000.0)) # m -> mm
        


    def rectification(self, left_img, right_img):
        # warp right image
        img_l = cv2.warpPerspective(left_img, self.H1, left_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)
        
        img_r = cv2.warpPerspective(right_img, self.H2, right_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)
        return img_l, img_r

    def create_disparity_map(self, left_img, right_img, is_rectify_enabled = True):

        # left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        # right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

        if is_rectify_enabled:
            left_img_rect, right_img_rect = self.rectification(left_img, right_img) # Rectification using Homography
            # print("rectified")
        else:
            left_img_rect = left_img
            right_img_rect = right_img

        self.disparity = self.stereoProcessor.compute(left_img_rect, right_img_rect)
        cv2.filterSpeckles(self.disparity, 0, 60, self.max_disparity)
        
        _, self.disparity = cv2.threshold(self.disparity, 0, self.max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (self.disparity / 16.).astype(np.uint8)
        # frame = cv2.applyColorMap(disparity_scaled, cv2.COLORMAP_HOT)
        # frame[frame > 200] = 0
        # cv2.imshow("Scaled Disparity stream", frame)

        disparity_colour_mapped = cv2.applyColorMap(
            (disparity_scaled * (256. / self.max_disparity)).astype(np.uint8),
            cv2.COLORMAP_HOT)
        cv2.imshow("Disparity", disparity_colour_mapped)
        cv2.imshow("rectified left", left_img_rect)
        cv2.imshow("rectified right", right_img_rect)


        key = cv2.waitKey(1)


        

        







