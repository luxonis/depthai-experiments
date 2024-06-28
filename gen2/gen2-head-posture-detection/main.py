
from depthai_sdk import OakCamera, TextPosition, Visualizer
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
import cv2
MIN_THRESHOLD = 15. # Degrees in yaw/pitch/roll to be considered as head movement

with OakCamera() as oak:
    color = oak.create_camera('color')
    det_nn = oak.create_nn('face-detection-retail-0004', color)
    # Passthrough is enabled for debugging purposes
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det_nn.config_nn(resize_mode='crop')

    headpose = oak.create_nn('head-pose-estimation-adas-0001', input=det_nn)
    # emotion_nn.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket):
        vis: Visualizer = packet.visualizer
        for det, rec in zip(packet.detections, packet.nnData):
            yaw = rec.getLayerFp16('angle_y_fc')[0]
            pitch = rec.getLayerFp16('angle_p_fc')[0]
            roll = rec.getLayerFp16('angle_r_fc')[0]
            print("pitch:{:.0f}, yaw:{:.0f}, roll:{:.0f}".format(pitch,yaw,roll))

            vals = np.array([abs(pitch),abs(yaw),abs(roll)])
            max_index = np.argmax(vals)

            if vals[max_index] < MIN_THRESHOLD: continue
            """
            pitch > 0 Head down, < 0 look up
            yaw > 0 Turn right < 0 Turn left
            roll > 0 Tilt right, < 0 Tilt left
            """
            if max_index == 0:
                if pitch > 0: txt = "Look down"
                else: txt = "Look up"
            elif max_index == 1:
                if yaw > 0: txt = "Turn right"
                else: txt = "Turn left"
            elif max_index == 2:
                if roll > 0: txt = "Tilt right"
                else: txt = "Tilt left"

            vis.add_text(txt,
                        bbox=(*det.top_left, *det.bottom_right),
                        position=TextPosition.BOTTOM_MID)

        vis.draw(packet.frame)
        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(headpose, callback=cb, fps=True)
    oak.visualize(det_nn.out.passthrough)
    oak.visualize(headpose.out.twostage_crops, scale=3.0)
    # oak.show_graph() # Show pipeline graph, no need for now
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)

