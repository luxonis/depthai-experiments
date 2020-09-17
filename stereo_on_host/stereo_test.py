from stereo_on_host import StereoSGBM
import cv2
import numpy as np
import glob



if __name__ == "__main__":

    left_images_paths = glob.glob("stereo_on_host/dataset/left" + '/left*.png')
    right_images_paths = glob.glob("stereo_on_host/dataset/right" + '/right*.png')
    H_right = np.array([[1.012611746788025, -0.009315459989011288, -16.0528621673584],
                        [0.017922570928931236, 1.0018900632858276, -16.368852615356445], 
                        [1.9619521481217816e-05, 2.267290710733505e-06, 0.9867464303970337]])

    # H_left = np.array([[1.0085175037384033, 0.02535983733832836, -24.814842224121094],
    #                     [-0.021759534254670143, 0.998845636844635, 15.92939281463623],
    #                     [1.4236070455808658e-05, -2.179780722144642e-06, 0.9917676448822021]])
    left_images_paths.sort()
    right_images_paths.sort()
    print(left_images_paths)
    stereo_process = StereoSGBM(7.5, H_right)

    for left_img_path, right_img_path in zip(left_images_paths, right_images_paths):
        stereo_process.create_disparity_map(left_img_path, right_img_path)
