import blobconverter
import cv2
import depthai as dai

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
cam.preview.link(image_manip_script.inputs['frame'])
image_manip_script.setScript("""
import time
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999
while True:
    frame = node.io['frame'].get()
    inDets = node.io['nn_in'].tryGet()
    if inDets is not None:
        # node.warn(f"People detected: {len(inDets.detections)}")
        for det in inDets.detections:
            limit_roi(det)
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(48, 96)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(frame)
""")

# Recieves 256 byte person vector, runs cosinus distance between them
reid_script = p.create(dai.node.Script)
reid_script.setProcessor(dai.ProcessorType.LEON_CSS)
with open("reid.py", "r") as f:
    reid_script.setScript(f.read())
# Forward person bounding box from one script to another
image_manip_script.outputs['manip_cfg'].link(reid_script.inputs['manip_cfg'])

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.
manip_crop = p.create(dai.node.ImageManip)
image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)

manip_crop.initialConfig.setResize(48, 96)
manip_crop.setWaitForConfigInput(True)

crop_xout = p.createXLinkOut()
crop_xout.setStreamName("crop")
manip_crop.out.link(crop_xout.input)

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

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    videoQ = device.getOutputQueue(name="video")
    cropQ = device.getOutputQueue(name="crop")
    # faceDetQ = device.getOutputQueue(name="face_det", maxSize=1, blocking=False)
    # nnQ = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        frame = videoQ.get().getCvFrame()
        cv2.imshow("frame", frame)

        if cropQ.has():
            cv2.imshow("crop", cropQ.get().getCvFrame())
        #     dets = faceDetQ.get().detections
        #     if 0 < len(dets):
        #         faceDet = dets[0]
        #         face_coords = [faceDet.xmin, faceDet.ymin,faceDet.xmax,faceDet.ymax]
        #         nndata = nnQ.get()


        if cv2.waitKey(1) == ord('q'):
            break