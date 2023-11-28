import cv2
import numpy as np
import argparse
import depthai as dai
import collections
import time
from itertools import cycle
from camera_properties import CameraResponse, Camera_creation, FPS
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print("DepthAI version:", dai.__version__)
print("DepthAI path:", dai.__file__)

device_info = {}
with dai.Device() as device:
    camera = Camera_creation(device)
    pipeline = camera.pipelineCreation()
    device.startPipeline(pipeline)
    response = CameraResponse(device)
    q = {}
    fps_host = {}  # FPS computed based on the time we receive frames in app
    fps_capt = {}  # FPS computed based on capture timestamps from device
    for c in camera.streams:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
        fps_host[c] = FPS()
        fps_capt[c] = FPS()
    capture_list = []
    end = False
    while True:
        for c in camera.streams:
            pkt = q[c].tryGet()
            if pkt is not None:
                fps_host[c].update()
                fps_capt[c].update(pkt.getTimestamp().total_seconds())
                frame = pkt.getCvFrame()
                capture = c in capture_list
                if capture:
                    capture_file_info = ('capture_' + c + '_' + camera.sensors[c] 
                         + '_' + str(frame.shape[0]) + 'x' + str(frame.shape[1])
                         + '_exp_' + str(int(pkt.getExposureTime().total_seconds()*1e6))
                         + '_iso_' + str(pkt.getSensitivity())
                         + '_lens_' + str(pkt.getLensPosition())
                         + '_' + capture_time
                         + '_' + str(pkt.getSequenceNum())
                        )
                    filename = capture_file_info + '.png'
                    import os
                    if not os.path.exists(f"{dir_path}/dataset/{time.strftime('%Y-%m-%d')}"):
                        os.makedirs(f"{dir_path}/dataset/{time.strftime('%Y-%m-%d')}")
                    print('Saving:', f"{dir_path}/dataset/{time.strftime('%Y-%m-%d')}/{filename}")
                    cv2.imwrite(f"{dir_path}/dataset/{time.strftime('%Y-%m-%d')}/{filename}", frame)
                    capture_list.remove(c)
                cv2.imshow(c, frame)
                key = cv2.waitKey(1)
                if key == ord ("c"):
                    capture_time = time.strftime('%Y%m%d_%H%M%S')
                    capture_list = camera.streams.copy()
                    print(f"Capture all images")
                print("\rFPS:",
                    *["{:6.2f}|{:6.2f}".format(fps_host[c].get(), fps_capt[c].get()) for c in camera.streams],
                    end=' ', flush=True)
                response.CameraControl(key,camera.streams)
                if key == ord("q"):
                    end = True
                    break
        if end:
            break
            



