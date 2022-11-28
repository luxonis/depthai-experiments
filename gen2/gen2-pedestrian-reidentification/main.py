#!/usr/bin/env python3

from depthai_sdk import Camera
import depthai as dai

with Camera() as cam:
    rgb_camera = cam.create_camera(out=True, name='cam')
    person_det = cam.create_nn("person-detection-retail-0013", rgb_camera, name="det", out=True)
    # If input from an object detection NN, crop around BB from HQ frames for new model
    reid = cam.create_nn("person-reidentification-retail-0288", input=person_det, name="reid", out=True)

    cam.start()
    
    def cb_reid(msgs):
        det = dai.ImgDetections = msgs['det']
        nndata: dai.NNData = msgs['reid']
        frame = dai.ImgFrame = msgs['cam']
        
        # ... Run cosine distance between reid vectors ...
        # ... Draw rectangles with ID (tracking) ...
        # ... Show frame with cv2.imshow ...

    cb = cam.create_msg_callback([rgb_camera, person_det, reid], cb_reid)

    while cam.running():
        msgs = cam.get_synced_msgs()
        cb.call(msgs)

        cam.poll()
