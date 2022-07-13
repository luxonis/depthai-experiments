#!/usr/bin/env python3

from depthai_sdk import Camera
import depthai as dai

with Camera(input='images/') as cam:
    rgb_camera = cam.create_camera(args=True, out=True) # args = True to enable option to stream from mp4
    person_det = cam.create_nn('person-detection-retail-0013', rgb_camera, out=True) # Path to json
    
    cam.start()
    
    visualizer = cam.create_visualizer(rgb_camera, person_det)

    num = ''
    def cb_calc(msg: dai.ImgDetections):
        global num
        # Calc whether person has gone up/down/left/right
        num = len(msg)

    def vis_callback(frame):
        global num
        # Add some overlay to the frame
        visualizer.print(f"Number of people: {num}") # Write some stuff to the frame
        visualizer.show(frame) # Now do the cv2.imshow

    cam.create_msg_callback(person_det, cb_calc)

    while cam.running():
        msgs = cam.get_synced_msgs()
        visualizer.visualize(msgs, vis_callback) # Don't show frame just yet

        cam.poll() # name tbd (does its GUI stuff)
