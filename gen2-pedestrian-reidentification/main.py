import blobconverter
import cv2
import depthai as dai
import json
import time
from utility import *

p = dai.Pipeline()

cam = p.create(dai.node.ColorCamera)
cam.setIspScale(2,3)
cam.setInterleaved(False)
cam.setVideoSize(1088,640)
cam.setPreviewSize(1088,640)

# Send color frames to the host via XLink
cam_xout = p.create(dai.node.XLinkOut)
cam_xout.setStreamName("video")
cam.video.link(cam_xout.input)

# Crop 1088x640 -> 544x320
people_manip = p.create(dai.node.ImageManip)
people_manip.initialConfig.setResize(544, 320)
cam.preview.link(people_manip.inputImage)

# NN that detects faces in the image
detection_nn = p.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setConfidenceThreshold(0.6)
detection_nn.setBlobPath(blobconverter.from_zoo("person-detection-retail-0013", shaves=6))
# Specify that network takes latest arriving frame in non-blocking manner
detection_nn.input.setQueueSize(4)
detection_nn.input.setBlocking(False)
people_manip.out.link(detection_nn.input)

# Script node will take the output from the NN as an input, get the first bounding box
# and send ImageManipConfig to the manip_crop
image_manip_script = p.create(dai.node.Script)
detection_nn.out.link(image_manip_script.inputs['nn_in'])
detection_nn.passthrough.link(image_manip_script.inputs['passthrough'])
cam.preview.link(image_manip_script.inputs['frame'])

with open("script-crop.py", "r") as f:
    image_manip_script.setScript(f.read())

# Recieves 256 byte person vector, runs cosinus distance between them
reid_script = p.create(dai.node.Script)
with open("script-reid.py", "r") as f:
    reid_script.setScript(f.read())
# Forward person bounding box from one script to another
image_manip_script.outputs['manip_cfg'].link(reid_script.inputs['cfg'])

# This ImageManip will crop the frame based on the NN detections. Resulting image will be the cropped
manip_crop = p.create(dai.node.ImageManip)
image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)

manip_crop.initialConfig.setResize(48, 96)
manip_crop.setWaitForConfigInput(True)

# Second NN that detcts emotions from the cropped 64x64 face
reid_nn = p.createNeuralNetwork()
reid_nn.setBlobPath("models/person-reidentification-retail-0031_96x48.blob")
# landmarks_nn.setBlobPath(blobconverter.from_zoo(name="facial_landmarks_68_160x160", zoo_type="depthai", version=openvinoVersion))
# reid_nn.setBlobPath(blobconverter.from_zoo(
#     name="person-reidentification-retail-0031_96x48",
#     zoo_type="depthai",
#     use_cache=False))
manip_crop.out.link(reid_nn.input)
reid_nn.out.link(reid_script.inputs['reid'])

# Compare vectors by cos similarity
cos_nn = p.createNeuralNetwork()
cos_nn.setBlobPath("models/cos_dist_simplified_openvino_2021.4_6shave.blob")
reid_script.outputs['a'].link(cos_nn.inputs['a'])
reid_script.outputs['b'].link(cos_nn.inputs['b'])
cos_nn.out.link(reid_script.inputs['cos'])

result_xout = p.createXLinkOut()
result_xout.setStreamName("out")
reid_script.outputs['out'].link(result_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    videoQ = device.getOutputQueue(name="video")
    outQ = device.getOutputQueue(name="out")
    results = {}
    text = TextHelper()
    while True:
        frame = videoQ.get().getCvFrame()

        if outQ.has():
            jsonText = str(outQ.get().getData(), "ascii")
            dict = json.loads(jsonText)
            results[str(dict['id'])] = {
                'ts': time.time(),
                'bbox': dict['bb'],
                'cnt': dict['cnt'],
            }

        for id, obj in results.items():
            # print(id, obj)
            # Remove results older than 0.2 sec
            if time.time() - obj['ts'] > 0.2:
                # results.pop(id)
                continue
            bbox = frameNorm(frame, obj['bbox'])
            text.putText(frame, f"ID: {id}", (bbox[0] + 10, bbox[1] + 30))
            text.putText(frame, f"Age: {obj['cnt']}", (bbox[0] + 10, bbox[1] + 60))
            text.rectangle(frame, bbox)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break