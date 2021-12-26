# coding=utf-8

'''
This demo consists of a pipeline that performs face recognition on the camera input with little
host side processing (almost all on device using script node)

The pipeline, in general, does the following (not all nodes are described):
1) In ImageManip node resizes the camera preview output and feeds them to a face detection NN.
1) A face detection NN generates an ImgDetections from frames and sends them to the script node.
2) The ImgDections contains a list of all faces detected.  For each detection, which consists
   of a bounding box, the script node creates an ImageManipConfig to crop out the faces, 
   resize them, and change the output format.  This is sent to an ImageManip node. Note: one
   configuration per face is sent (and there may be more than one face detected per image).
3) An ImageManip node performs the crop/resize/format change and sends each face to a headpose
   estimator NN.
4) The headpose estimator NN calculates the headpose and sends them to the script node.
5) The script node reads (only) the "rotation" value from the headpose estimate and calculates
   a rotated rectangle for cropping that will work better with the face recognition node.  
   The script creates an ImangeManipConfig with this rotated rectangle and sends to an
   ImageManip node.
6) An ImageManip node performs the crop and resize and sends each face to a face recognition NN.
4) The face recognition NN generates features based upon the face image sent.

The pipeline outputs (xlinks) the camera preview output, the face detections, the rotated 
rectangle values, and the face recognition features.

Its important to note that there is a 1:N relationship between face detections and the latter
outputs of the pipeline.  For each face detection (i.e. ImgDetection) with N number of faces
detected, there are N number of rotated rectangle values and face recognition features.  

This pipeline works because all node inputs are blocking and therefore, remain can remain in sync.
However, if the host does not process all of the output queues fast enough, its possible that
a node's output queue overruns and something will be dropped.  When this happens, the pipeline
can become out of sync and recognitions may not line up with their rotated rectangles.

This demo was tested on an OAK-D with a 5 FPS and two people in the database.  At a FPS of 10, there
were errors in the recognition and its suspected that one output queue might have overflowed.

An option is to create threads where one thread reads the output from the pipeline and stores it in
a Queue.  The second thread reads this Queue and outputs to the display.  This way you may miss
a recognition that that queue overflows because the frame rate is fast, but you don't overrun the
output queue of a node and lose sync (if this is what is really happening).


''' 
import os
from datetime import timedelta, datetime
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

DISPLAY_FACE = False
VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

class HostSync:
    def __init__(self):
        self.array = []
    def add_msg(self, msg):
        self.array.append(msg)
    def get_msg(self, timestamp):
        def getDiff(msg, timestamp):
            return abs(msg.getTimestamp() - timestamp)
        if len(self.array) == 0: return None

        self.array.sort(key=lambda msg: getDiff(msg, timestamp))

        # Remove all frames that are older than 0.5 sec
        for i in range(len(self.array)):
            j = len(self.array) - 1 - i
            if getDiff(self.array[j], timestamp) > timedelta(milliseconds=500):
                self.array.remove(self.array[j])

        if len(self.array) == 0: return None
        return self.array.pop(0)

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)
    def drawContours(self, frame, points):
        cv2.drawContours(frame, [points], 0, self.bg_color, 6)
        cv2.drawContours(frame, [points], 0, self.color, 2)

class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{databases}/{self.name}", *db_)
        self.adding_new = False

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)
    openvino_version = '2021.2'

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    # For ImageManip rotate you need input frame of multiple of 16
    cam.setPreviewSize(1072, 1072)
    cam.setVideoSize(VIDEO_SIZE)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setFps(5)

    host_face_out = pipeline.create(dai.node.XLinkOut)
    host_face_out.setStreamName('frame')
    cam.video.link(host_face_out.input)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    # NeuralNetwork
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

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    script = pipeline.create(dai.node.Script)

    face_det_nn.out.link(script.inputs['face_det_in'])
    
    face_detections_out = pipeline.create(dai.node.XLinkOut)
    face_detections_out.setStreamName('detections')
    face_det_nn.out.link(face_detections_out.input)

    # We are only interested in timestamp, so we can sync depth frames with NN output
    face_det_nn.passthrough.link(script.inputs['face_pass'])

    with open("script.py", "r") as f:
        script.setScript(f.read())

    # ImageManip as a workaround to have more frames in the pool.
    # cam.preview can only have 4 frames in the pool before it will
    # wait (freeze). Copying frames and setting ImageManip pool size to
    # higher number will fix this issue.
    copy_manip = pipeline.create(dai.node.ImageManip)
    cam.preview.link(copy_manip.inputImage)
    copy_manip.setNumFramesPool(20)
    copy_manip.setMaxOutputFrameSize(1072*1072*3)

    copy_manip.out.link(face_det_manip.inputImage)
    copy_manip.out.link(script.inputs['preview'])

    print("Creating Head pose estimation NN")
    headpose_manip = pipeline.create(dai.node.ImageManip)
    headpose_manip.setWaitForConfigInput(True) # needed to maintain sync
    headpose_manip.initialConfig.setResize(60, 60)

    script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
    script.outputs['manip_img'].link(headpose_manip.inputImage)

    headpose_nn = pipeline.create(dai.node.NeuralNetwork)
    headpose_nn.setBlobPath(blobconverter.from_zoo(
        name="head-pose-estimation-adas-0001",
        shaves=6,
        version=openvino_version
    ))
    headpose_manip.out.link(headpose_nn.input)

    headpose_nn.out.link(script.inputs['headpose_in'])
    headpose_nn.passthrough.link(script.inputs['headpose_pass'])

    print("Creating face recognition ImageManip/NN")

    face_rec_manip = pipeline.create(dai.node.ImageManip)
    face_rec_manip.setWaitForConfigInput(True) # needed to maintain sync
    face_rec_manip.initialConfig.setResize(112, 112)

    script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
    script.outputs['manip2_img'].link(face_rec_manip.inputImage)

    face_rec_cfg_out = pipeline.create(dai.node.XLinkOut)
    face_rec_cfg_out.setStreamName('face_rec_cfg_out')
    script.outputs['manip2_cfg'].link(face_rec_cfg_out.input)

    # Only send metadata for the host-side sync
    # pass2_out = pipeline.create(dai.node.XLinkOut)
    # pass2_out.setStreamName('pass2')
    # pass2_out.setMetadataOnly(True)
    # script.outputs['manip2_img'].link(pass2_out.input)

    face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
    # Removed from OMZ, so we can't use blobconverter for downloading, see here:
    # https://github.com/openvinotoolkit/open_model_zoo/issues/2448#issuecomment-851435301
    face_rec_nn.setBlobPath("models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
    face_rec_manip.out.link(face_rec_nn.input)

    if DISPLAY_FACE:
        xout_face = pipeline.createXLinkOut()
        xout_face.setStreamName('face')
        face_rec_manip.out.link(xout_face.input)

    arc_out = pipeline.create(dai.node.XLinkOut)
    arc_out.setStreamName('arc_out')
    face_rec_nn.out.link(arc_out.input)

    return pipeline


with dai.Device(create_pipeline()) as device:
    frameQ = device.getOutputQueue("frame", 15, False)
    recCfgQ = device.getOutputQueue("face_rec_cfg_out", 15, False)
    arcQ = device.getOutputQueue("arc_out", 15, False)
    detQ = device.getOutputQueue("detections", 15, False)
    if DISPLAY_FACE:
        faceQ = device.getOutputQueue("face", 15, False)

    facerec = FaceRecognition(databases, args.name)
    sync = HostSync()
    text = TextHelper()

    frameArr = []
    frame = None
    results = {}
    frameDelay = 0 #Note: was 6.  See comment below
    while True:
        frameIn = frameQ.tryGet()
        if frameIn is not None:
            frameArr.append(frameIn.getCvFrame())
            # The original version of this demo delayed the video ouput to sync the detections
            # with the video (head pose estimation and face recognition takes about ~200ms).
            # However, the frame rate was slowed down to avoid what I suspect are drops in the
            # output queue (which I believe lead to sync issues).
            # At the current frame rate (5 fps), 0 frame delay appears to work well enough.
            if frameDelay < len(frameArr):
                frame = frameArr.pop(0)

        inDet = detQ.tryGet()
        if inDet is not None:
            for det in inDet.detections:
                cfg = recCfgQ.get()
                arcIn = arcQ.get()
                rr = cfg.getRaw().cropConfig.cropRotatedRect
                h, w = VIDEO_SIZE
                center = (int(rr.center.x * w), int(rr.center.y * h))
                size = (int(rr.size.width * w), int(rr.size.height * h))
                rotatedRect = (center, size, rr.angle)
                points = np.int0(cv2.boxPoints(rotatedRect))
                # draw_detections(frame, face_in.detections)
                features = np.array(arcIn.getFirstLayerFp16())
                # print('New features')
                conf, name = facerec.new_recognition(features)
                results[name] = {
                    'conf': conf,
                    'coords': center,
                    'points': points,
                    'ts': time.time()
                }

        if frame is not None:
            for name, result in results.items():
                if time.time() - result["ts"] < 0.15:
                    text.drawContours(frame, result['points'])
                    text.putText(frame, f"{name} {(100*result['conf']):.0f}%", result['coords'])

            cv2.imshow("color", cv2.resize(frame, (800,800)))

        if DISPLAY_FACE and faceQ.has():
            cv2.imshow('face', faceQ.get().getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
