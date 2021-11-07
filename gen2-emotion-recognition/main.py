import blobconverter
import numpy as np
import cv2
import depthai as dai

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)

class_names = ['neutral', 'happy', 'sad', 'surprise', 'anger']

openvinoVersion = "2021.3"
p = dai.Pipeline()
p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)

cam = p.create(dai.node.ColorCamera)
cam.setIspScale(2,3)
cam.setInterleaved(False)
cam.setVideoSize(720,720)
cam.setPreviewSize(720,720)

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
face_nn.setBlobPath(str(blobconverter.from_zoo("face-detection-retail-0004", shaves=6, version=openvinoVersion)))
face_det_manip.out.link(face_nn.input)

# Send ImageManipConfig to host so it can visualize the landmarks
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
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    frame = node.io['frame'].get()
    face_dets = node.io['nn_in'].get().detections

    # node.warn(f"Faces detected: {len(face_dets)}")
    for det in face_dets:
        limit_roi(det)
        # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
        cfg = ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        cfg.setResize(64, 64)
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
manip_crop.initialConfig.setResize(64, 64)
manip_crop.setWaitForConfigInput(True)

# Second NN that detcts emotions from the cropped 64x64 face
landmarks_nn = p.createNeuralNetwork()
landmarks_nn.setBlobPath(str(blobconverter.from_zoo("emotions-recognition-retail-0003", shaves=6, version=openvinoVersion)))
manip_crop.out.link(landmarks_nn.input)

landmarks_nn_xout = p.createXLinkOut()
landmarks_nn_xout.setStreamName("emotions")
landmarks_nn.out.link(landmarks_nn_xout.input)

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    videoQ = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    faceDetQ = device.getOutputQueue(name="face_det", maxSize=4, blocking=False)
    emotionsQ = device.getOutputQueue(name="emotions", maxSize=4, blocking=False)

    textHelper = TextHelper()
    while True:
        if faceDetQ.has():
            detections = faceDetQ.get().detections
            frame = videoQ.get().getCvFrame()
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
                # Each face detection will be sent to emotion estimation model. Wait for the result
                nndata = emotionsQ.get()
                # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in nndata.getAllLayers()]

                results = np.array(nndata.getFirstLayerFp16())
                result_conf = np.max(results)
                if 0.3 < result_conf:
                    name = class_names[np.argmax(results)]
                    conf = round(100 * result_conf, 1)
                    textHelper.putText(frame, f"{name}, {conf}%", (bbox[0] + 10, bbox[1] + 20))

            cv2.imshow("frame", frame)
        # if frame is not None:

        if cv2.waitKey(1) == ord('q'):
            break
