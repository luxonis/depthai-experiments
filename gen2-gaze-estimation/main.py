# coding=utf-8
import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
from MultiMsgSync import TwoStageHostSeqSync

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

VIDEO_SIZE = (1072, 1072)

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
openvino_version = '2021.4'

cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(1072, 1072)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

host_face_out = pipeline.create(dai.node.XLinkOut)
host_face_out.setStreamName('color')
cam.video.link(host_face_out.input)

# ImageManip as a workaround to have more frames in the pool.
# cam.preview can only have 4 frames in the pool before it will
# wait (freeze). Copying frames and setting ImageManip pool size to
# higher number will fix this issue.
copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(1072*1072*3)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
face_det_manip.setMaxOutputFrameSize(300*300*3)
copy_manip.out.link(face_det_manip.inputImage)

#=================[ FACE DETECTION ]=================

print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(
    name="face-detection-retail-0004",
    shaves=6,
    version=openvino_version
))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

#=================[ SCRIPT NODE ]=================

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
face_det_nn.passthrough.link(script.inputs['face_pass'])

copy_manip.out.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

#=================[ HEAD POSE ESTIMATION ]=================

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.inputConfig.setWaitForMessage(True)
script.outputs['headpose_cfg'].link(headpose_manip.inputConfig)
script.outputs['headpose_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    shaves=6,
    version=openvino_version
))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

#=================[ LANDMARKS DETECTION ]=================

landmark_manip = pipeline.create(dai.node.ImageManip)
landmark_manip.initialConfig.setResize(48, 48)
landmark_manip.inputConfig.setWaitForMessage(True)
script.outputs['landmark_cfg'].link(landmark_manip.inputConfig)
script.outputs['landmark_img'].link(landmark_manip.inputImage)

landmark_nn = pipeline.create(dai.node.NeuralNetwork)
landmark_nn.setBlobPath(blobconverter.from_zoo(
    name="landmarks-regression-retail-0009",
    shaves=6,
    version=openvino_version
))
landmark_manip.out.link(landmark_nn.input)

landmark_nn.out.link(script.inputs['landmark_in'])
landmark_nn.passthrough.link(script.inputs['landmark_pass'])

landmark_xout = pipeline.create(dai.node.XLinkOut)
landmark_xout.setStreamName('landmarks')
landmark_nn.out.link(landmark_xout.input)


#=================[ LEFT EYE CROP ]=================

left_manip = pipeline.create(dai.node.ImageManip)
left_manip.initialConfig.setResize(60, 60)
left_manip.inputConfig.setWaitForMessage(True)
script.outputs['left_manip_img'].link(left_manip.inputImage)
script.outputs['left_manip_cfg'].link(left_manip.inputConfig)
left_manip.out.link(script.inputs['left_eye_in'])

# m1 = pipeline.create(dai.node.XLinkOut)
# m1.setStreamName("m1")
# left_manip.out.link(m1.input)


#=================[ LEFT EYE CROP ]=================

right_manip = pipeline.create(dai.node.ImageManip)
right_manip.initialConfig.setResize(60, 60)
right_manip.inputConfig.setWaitForMessage(True)
script.outputs['right_manip_img'].link(right_manip.inputImage)
script.outputs['right_manip_cfg'].link(right_manip.inputConfig)
right_manip.out.link(script.inputs['right_eye_in'])

# m2 = pipeline.create(dai.node.XLinkOut)
# m2.setStreamName("m2")
# right_manip.out.link(m2.input)


#=================[ GAZE ESTIMATION ]=================


gaze_nn = pipeline.create(dai.node.NeuralNetwork)
gaze_nn.setBlobPath(blobconverter.from_zoo(
    name="gaze-estimation-adas-0002",
    shaves=6,
    version=openvino_version
))
script.outputs['to_gaze'].link(gaze_nn.input)

gaze_xout = pipeline.create(dai.node.XLinkOut)
gaze_xout.setStreamName('recognition')
gaze_nn.out.link(gaze_xout.input)

#==================================================

with dai.Device(pipeline) as device:
    sync = TwoStageHostSeqSync()
    text = TextHelper()

    queues = {}
    # Create output queues
    for name in ["color", "detection", "landmarks", "recognition"]: #, "m1", "m2"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
            if q.has():
                # print(f"got msg {name}")
                msg = q.get()
                sync.add_msg(msg, name)
                if name == "color":
                    cv2.imshow("video", msg.getCvFrame())

        msgs = sync.get_msgs()
        if msgs is not None:
            # print("synced")
            cv2.imshow("ASD", msgs["color"].getCvFrame())
            dets = msgs["detection"].detections
            # for i, detection in enumerate(dets):
                # bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                # landmarks = np.array(msgs["landmarks"][0].getFirstLayerFp16())
                # text.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10,bbox[1] + 35))
                # break


        if cv2.waitKey(1) == ord('q'):
            break
