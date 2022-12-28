from depthai_sdk import OakCamera, TwoStagePacket, Visualizer, TextPosition
import numpy as np
import cv2

with OakCamera() as oak:
    color = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', color)
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det.config_nn(resize_mode='crop')

    age_gender = oak.create_nn('age-gender-recognition-retail-0013', input=det)

    def cb(packet: TwoStagePacket, visualizer: Visualizer):
        for det, rec in zip(packet.detections, packet.nnData):
            age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
            gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
            gender_str = "woman" if gender[0] > gender[1] else "man"

            visualizer.add_text(f'{gender_str}\nage: {age}',
                                bbox=(*det.top_left, *det.bottom_right),
                                position=TextPosition.BOTTOM_RIGHT)

        frame = visualizer.draw(packet.frame)
        cv2.imshow('Age-gender estimation', frame)


    # Visualize detections on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(age_gender, callback=cb).detections(fill_transparency=0.1)
    oak.visualize(det.out.passthrough)
    oak.visualize(age_gender.out.twostage_crops, scale=3.0)


    # oak.show_graph() # Show pipeline graph
    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)
