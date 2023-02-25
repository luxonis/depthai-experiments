import blobconverter
import cv2
import depthai as dai

import face_landmarks

p = dai.Pipeline()

cam = p.create(dai.node.ColorCamera)
# cam.setIspScale(2,3)
cam.setInterleaved(False)
cam.setVideoSize(1080, 1080)
cam.setPreviewSize(1080, 1080)

# Send color frames to the host via XLink
cam_xout = p.create(dai.node.XLinkOut)
cam_xout.setStreamName("video")
cam.video.link(cam_xout.input)

# Crop 720x720 -> 300x300
face_det_manip = p.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
cam.preview.link(face_det_manip.inputImage)

# NN that detects faces in the image
face_nn = p.create(dai.node.MobileNetDetectionNetwork)
face_nn.setConfidenceThreshold(0.3)
face_nn.setBlobPath(blobconverter.from_zoo("face-detection-retail-0004", shaves=6))
face_det_manip.out.link(face_nn.input)

# # Send ImageManipConfig to host so it can visualize the landmarks
config_xout = p.create(dai.node.XLinkOut)
config_xout.setStreamName("face_det")
face_nn.out.link(config_xout.input)

# Script node will take the output from the NN as an input, get the first bounding box
# and send ImageManipConfig to the manip_crop
image_manip_script = p.create(dai.node.Script)
face_nn.out.link(image_manip_script.inputs['nn_in'])
cam.preview.link(image_manip_script.inputs['frame'])
image_manip_script.setScript("""
import time
def enlrage_roi(det): # For better face landmarks NN results
    det.xmin -= 0.05
    det.ymin -= 0.02
    det.xmax += 0.05
    det.ymax += 0.02
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    frame = node.io['frame'].get()
    face_dets = node.io['nn_in'].get().detections

    # No faces found
    if len(face_dets) == 0: continue

    # Take the first detected face, since this demo isn't multiple-face fatigue detection
    det = face_dets[0]
    enlrage_roi(det)
    limit_roi(det)

    # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
    cfg = ImageManipConfig()
    cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
    cfg.setResize(160, 160)
    cfg.setKeepAspectRatio(False)
    node.io['manip_cfg'].send(cfg)
    node.io['manip_img'].send(frame)
    # node.warn(f"1 from nn_in: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
""")

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.
manip_crop = p.create(dai.node.ImageManip)
image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)
manip_crop.initialConfig.setResize(160, 160)
manip_crop.inputConfig.setWaitForMessage(True)

# Second NN that detcts emotions from the cropped 64x64 face
landmarks_nn = p.create(dai.node.NeuralNetwork)
landmarks_nn.setBlobPath(blobconverter.from_zoo(
    name="facial_landmarks_68_160x160",
    shaves=6,
    zoo_type="depthai",
))
manip_crop.out.link(landmarks_nn.input)

landmarks_nn_xout = p.create(dai.node.XLinkOut)
landmarks_nn_xout.setStreamName("nn")
landmarks_nn.out.link(landmarks_nn_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    videoQ = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    faceDetQ = device.getOutputQueue(name="face_det", maxSize=1, blocking=False)
    nnQ = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    decode = face_landmarks.FaceLandmarks()

    while True:
        frame = videoQ.get().getCvFrame()
        frame = cv2.resize(frame, (720, 720))

        if nnQ.has():
            dets = faceDetQ.get().detections
            if 0 < len(dets):
                faceDet = dets[0]
                face_coords = [faceDet.xmin, faceDet.ymin, faceDet.xmax, faceDet.ymax]
                nndata = nnQ.get()
                decode.run_land68(frame, nndata, face_coords)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
