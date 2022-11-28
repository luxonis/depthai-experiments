#!/usr/bin/env python3

from depthai_sdk import Camera
import depthai as dai

with Camera() as cam:
    rgb_camera = cam.create_camera(args=True, out=True) # args = True to enable option to stream from mp4
    person_tracker = cam.create_nn('person-detection-retail-0013', rgb_camera, tracker=True, out=True) # Path to json
    person_tracker.config_tracker([1], type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    
    cam.start()
    
    visualizer = cam.create_visualizer(rgb_camera, person_tracker)

    data = ''
    def calc_movement(msg: dai.Tracklets):
        global data
        # Calc whether person has gone up/down/left/right
        up, down, left, right = utils.calculate_tracklet_movement(msg)
        data = f"Up: {up}, Down: {down}"

    def vis_callback(frame):
        global data
        # Add some overlay to the frame
        visualizer.print(data) # Write some stuff to the frame
        visualizer.show(frame) # Now do the cv2.imshow

    cam.create_msg_callback(person_tracker, calc_movement)

    while cam.running():
        msgs = cam.get_synced_msgs()
        visualizer.visualize(msgs, vis_callback) # Don't show frame just yet

        cam.poll() # name tbd (does its GUI stuff)
