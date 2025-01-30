import cv2
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket
import numpy as np
import depthai as dai

roi = None
new_roi = False
selecting_roi = False
def roi_selection(event, x, y, flags, param):
    global start_pts, selecting_roi, roi, new_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        start_pts = [x, y]

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            start_pts[2:] = [x, y]
            roi = start_pts

    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        new_roi = True
        if abs(start_pts[0] - x) < 100 or abs(start_pts[1] - y) < 100:
            roi = None
            return
        start_pts[2:] = [x, y]
        roi = start_pts


shape = None
def cb(packet: DetectionPacket):
    global roi, shape
    frame = packet.visualizer.draw(packet.frame)
    if roi is not None:
        x_start, y_start, x_end, y_end = roi
        # frame_roi = frame[y_start:y_end, x_start:x_end]

        # Create a mask with the same size as the frame
        mask = np.zeros_like(frame)
        mask[y_start:y_end, x_start:x_end] = frame[y_start:y_end, x_start:x_end]

        # Darken the unselected region
        frame_darkened = frame * 0.3
        frame_darkened = frame_darkened.astype(np.uint8)

        # Blend the mask with the darkened frame
        frame = cv2.addWeighted(frame_darkened, 1, mask, 1, 0)
    shape = frame.shape
    cv2.imshow('color', frame)

with OakCamera() as oak:
    cv2.namedWindow("color")
    cv2.setMouseCallback("color", roi_selection)

    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    oak.visualize(color, fps=True, callback=cb)
    oak.visualize([nn.out.passthrough, nn], fps=True, scale=2/3)

    manipIn = oak.pipeline.create(dai.node.XLinkIn)
    manipIn.setStreamName('manip_in')
    manipIn.out.link(nn.image_manip.inputConfig)

    oak.start() # Initialize the device and start the pipeline

    raw_cfg = nn.image_manip.initialConfig.get() # To keep existing settings

    manipQ = oak.device.getInputQueue('manip_in')

    while oak.running():
        oak.poll()
        if new_roi:
            new_roi = False
            conf = dai.ImageManipConfig()
            conf.set(raw_cfg)
            if roi is not None:
                normalized_roi = (roi[0]/shape[1], roi[1]/shape[0], roi[2]/shape[1], roi[3]/shape[0])
                conf.setCropRect(normalized_roi)
            manipQ.send(conf)
            print(roi)

cv2.destroyAllWindows()
