#!/usr/bin/env python3

from depthai_sdk import Camera
import depthai as dai

with Camera() as cam:
    rgb_camera = cam.create_camera(out=True)
    
    stereo = cam.create_stereo(resolution="720P")

    # Decoding & Spatial calc will be done on host, as model decoding isn't supported on device
    palm_det = cam.create_nn("palm_detection_128x128", name='palm', input=[rgb_camera, stereo], spatials=True, out=True)
    
    mobilenet = cam.create_nn("mobilenet-ssd", name='mobilenet', input=[rgb_camera, stereo], spatials=True, out=True)
    mobilenet.configSpatialDetector(threshLow=1000, threshHigh=20000, bbscale=0.2)

    cam.start()
    
    vis = cam.create_visualizer([rgb_camera, palm_det, mobilenet])

    def calc(results: dict):
        palmSpatials: dai.SpatialImgDetections = results['palm']
        ssdSpatials: dai.SpatialImgDetections = results['mobilenet']

        # Calculate distance between these 2 objects for the machine-safety
        
    cb = cam.create_msg_callback([palm_det, mobilenet], calc)

    while cam.running():
        msgs = cam.get_synced_msgs()
        cb.call(msgs) # Will filter out relevant msgs (from palm_det and mobilenet components) and call calc() function
        vis.visualize(msgs)

        cam.poll()

