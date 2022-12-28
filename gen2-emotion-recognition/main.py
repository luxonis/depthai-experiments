from depthai_sdk import OakCamera, TwoStagePacket, TextPosition
import numpy as np
import cv2

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

with OakCamera() as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('face-detection-retail-0004', color)
    # Passthrough is enabled for debugging purposes
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det.config_nn(resize_mode='crop')

    emotion_nn = oak.create_nn('emotions-recognition-retail-0003', input=det)
    # emotion_nn.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket, visualizer):
        for det, rec in zip(packet.detections, packet.nnData):
            emotion_results = np.array(rec.getFirstLayerFp16())
            emotion_name = emotions[np.argmax(emotion_results)]

            visualizer.add_text(emotion_name,
                                bbox=(*det.top_left, *det.bottom_right),
                                position=TextPosition.BOTTOM_RIGHT)

        visualizer.draw(packet.frame)
        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(emotion_nn, callback=cb, fps=True)
    oak.visualize(det.out.passthrough)
    # oak.show_graph() # Show pipeline graph
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)
