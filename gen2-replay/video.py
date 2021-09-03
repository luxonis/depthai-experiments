import cv2
import glob

IMG_SAVE_PATH = "/frames"

streams = ["depth", "right", "rgb", "left", "birdsview"]
for name in streams:
    if name == "birdsview": frameSize = (100, 300)
    else: frameSize = (672, 384)
    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

    for filename in sorted(glob.glob(f'{IMG_SAVE_PATH}/{name}/*.png')):
        img = cv2.imread(filename)
        print(filename)
        out.write(img)

    out.release()