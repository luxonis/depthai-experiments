import math

IMG_WIDTH = 1280
HFOV = 127 # Rectified depth degrees
BASELINE = 7.5

maxD = (BASELINE / 2) * math.tan((90 - HFOV / IMG_WIDTH)*math.pi/180)
print(f"Max depth perception: {maxD} mm")