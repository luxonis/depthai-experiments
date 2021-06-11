import cv2
import numpy as np
import glob
import re
cv2.namedWindow('comb2', cv2.WINDOW_NORMAL)

for cam in glob.glob('*/*'):
    print(cam)
    f_disp  = glob.glob(cam + '/*disparity*')[0]
    f_left  = glob.glob(cam + '/*left*')[0]
    f_right = glob.glob(cam + '/*right*')[0]
    f_rgb   = glob.glob(cam + '/*isp_4056*')[0]
    #print(f_disp, f_left, f_right, f_rgb)
    disp = cv2.imread(f_disp)
    left = cv2.imread(f_left)
    right = cv2.imread(f_right)
    rgb = cv2.imread(f_rgb)
    rgb = cv2.resize(rgb, (3202,2400), interpolation = cv2.INTER_AREA)
    #print(disp.shape)
    #black = np.zeros((640,1280,3)).astype(np.uint8)
    #comb = np.vstack((disp, left, right, black))
    comb = np.vstack((disp, left, right))
    #cv2.imshow('comb', comb)
    comb2 = np.hstack((rgb, comb))
    res = re.split(', |/|-|!', cam)
    device = res[0]
    iteration = res[1]
    focus = 'autofocus'
    if len(res) > 2: focus = res[2]
    name = device + '_' + 'capture' + iteration + '_' + focus + '.jpg'
    print(name)
    cv2.imwrite('Zoutput/'+name, comb2)
    cv2.imshow('comb2', comb2)
    cv2.waitKey(40)





