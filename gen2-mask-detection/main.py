from depthai_sdk import OakCamera, TwoStagePacket, AspectRatioResizeMode, Visualizer, TextPosition, BboxStyle
import numpy as np
import cv2

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

with OakCamera() as oak:
    color = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', color)
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det.config_nn(aspect_ratio_resize_mode=AspectRatioResizeMode.CROP)

    mask = oak.create_nn('sbd_mask_classification_224x224', input=det)
    # mask.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket, visualizer: Visualizer):
        for det, rec in zip(packet.detections, packet.nnData):
            index = np.argmax(log_softmax(rec.getFirstLayerFp16()))
            text = "No Mask"
            color = (0,0,255) # Red
            if index == 1:
                text = "Mask"
                color = (0,255,0)

            visualizer.add_text(text,
                                bbox=(*det.top_left, *det.bottom_right),
                                position=TextPosition.BOTTOM_MID)

        frame = visualizer.draw(packet.frame)
        cv2.imshow('Mask detection', frame)

    # Visualize detections on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(mask, callback=cb).detections(fill_transparency=0.1)
    oak.visualize(det.out.passthrough)

    # oak.show_graph() # Show pipeline graph, no need for now
    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)
