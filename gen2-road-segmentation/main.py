from depthai_sdk import Camera

with Camera() as cam:
    rgb_camera = cam.create_camera('color', out=True)
    face_det = cam.create_nn('road-segmentation-adas-0001', rgb_camera, out=True) # Path to json
    
    cam.start()
    
    face_visualizer = cam.create_visualizer(rgb_camera, face_det)

    while cam.running():
        msgs = cam.get_synced_msgs()
        face_visualizer.visualize(msgs)

        cam.poll() # name tbd (does its GUI stuff)
