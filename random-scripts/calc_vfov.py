import math
import numpy as np

def calc_f(width, hfov):
    return width * 0.5 / math.tan(hfov * 0.5 * math.pi/180)

def calc_fov_D_H_V(f, w, h):
   return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))

def print_D_H_V(f,w,h1):
    d,h,v = calc_fov_D_H_V(f,w,h1)
    print(f"DFOV: {d}, HFOV: {h}, VFOV: {v}")

HORIZONTAL_PIXELS = 4056
VERTICAL_PIXEL = 3040
HFOV = 140

print_D_H_V(calc_f(HORIZONTAL_PIXELS, HFOV), HORIZONTAL_PIXELS, VERTICAL_PIXEL)