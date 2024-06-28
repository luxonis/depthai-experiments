import math

IMG_WIDTH = 1280
HFOV = 97 # Rectified depth degrees

flen = IMG_WIDTH*0.5/math.tan(HFOV * 0.5 * math.pi / 180)
print(f"Focal length: {flen} pixels")