import numpy as np
import cv2

#
#       3 + + + + + + + + 7
#       +\                +\          UP
#       + \               + \
#       +  \              +  \        |
#       +   4 + + + + + + + + 8       | y
#       +   +             +   +       |
#       +   +             +   +       |
#       +   +     (0)     +   +       .------- x
#       +   +             +   +        \
#       1 + + + + + + + + 5   +         \
#        \  +              \  +          \ z
#         \ +               \ +           \
#          \+                \+
#           2 + + + + + + + + 6


def draw_box(image, pts, dist = None, vol = None):

    overlay = image.copy()

    # draw bottom
    bottom_sel = [1,2,6,5]
    bottom_pts = pts[bottom_sel]
    bottom_pts = bottom_pts.reshape((-1, 1, 2))
    bottom_pts = bottom_pts.astype(np.int32)
    cv2.fillPoly(overlay, pts=[bottom_pts], color =(192, 192, 192))

    # draw green size
    lines_side = [(1, 4), (2, 3)]
    for line in lines_side:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]), int(pt0[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
        cv2.line(overlay, pt0, pt1, (0, 255,0))

    # draw blue lines
    lines = [(1, 2), (2, 4), (1, 3), (4, 3), (2, 6), (1, 5), (3, 7), (4, 8), (6, 8), (7, 8), (7, 5), (5, 6)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]), int(pt0[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
        cv2.line(overlay, pt0, pt1, (255,0,0))

    # draw top lines
    lines_top = [(3, 8), (4, 7)]
    for line in lines_top:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]), int(pt0[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
        cv2.line(overlay, pt0, pt1, (0,0,255))

    if dist is not None:
        # show fps
        color_font, color_bg = (255, 255, 255), (255, 0, 0)
        label_fps = "Distance: {:.2f} m".format(dist/1000)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)

        x,y = np.min(pts, axis = 0).astype(np.int32)
        y -= (h1 + 15)
        if y < 0:
            y = 0

        cv2.rectangle(overlay, (x-2, y-h1-2), (x + w1 + 6, y + 4), color_bg, -1)
        cv2.putText(overlay, label_fps, (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_font)
        cv2.line(overlay, (x,y+2), (x,y), color_bg)

    if vol is not None:
        color_font, color_bg = (255, 255, 255), (0, 0, 255)
        vol_label = "Volume: {:.2f} m^3".format(vol)
        (w1, h1), _ = cv2.getTextSize(vol_label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
        x = int(pts[0,0])-50
        y = int(pts[0,1])
        cv2.rectangle(overlay, (x-2, y-h1-2), (x + w1 + 6, y + 4), color_bg, -1)
        cv2.putText(overlay, vol_label, (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_font)

    cv2.addWeighted(overlay, 0.5, image, 0.5,
                    0, image)
