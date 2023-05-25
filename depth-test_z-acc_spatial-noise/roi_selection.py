import cv2
import argparse
import numpy as np

roi_pts = []
roi_selected = False
live_selection = False


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file')
parser.add_argument('--outROI', type=str, required=True, help='Path to the output ROI file')
args = parser.parse_args()



def select_roi(event, x, y, flags, param):
    global roi_pts, roi_selected, live_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
        live_selection = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if live_selection:
            roi_pts[1:] = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts[1:] = [(x, y)]
        live_selection = False
        roi_selected = True




if args.input.endswith(('.mp4', '.avi')):
    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    _, frame = cap.read()
    cap.release()
else:
    frame = cv2.imread(args.input)
original_frame = frame.copy()

cv2.namedWindow('Input')
cv2.setMouseCallback('Input', select_roi)

while True:
    frame = original_frame.copy()
    if len(roi_pts) == 2:
        cv2.rectangle(frame, roi_pts[0], roi_pts[1], (0, 255, 0), 2)

    cv2.imshow('Input', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("ROI is {}".format(roi_pts))
        with open(args.outROI, "w") as f:
            ROI = (roi_pts[0][0], roi_pts[0][1], roi_pts[1][0], roi_pts[1][1])
            f.write(str(ROI))
        break

cv2.destroyAllWindows()