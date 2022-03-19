from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBoundingBoxScaleFactor(0.8)
    face_det_nn.setDepthLowerThreshold(100)
    face_det_nn.setDepthUpperThreshold(5000)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))

    cam.preview.link(face_det_manip.inputImage)
    stereo.depth.link(face_det_nn.inputDepth)

    # Link Face ImageManip -> Face detection NN node
    face_det_manip.out.link(face_det_nn.input)

    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])

    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])

    image_manip_script.setScript("""
l = [] # List of images
# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def get_latest_frame(seq):
    global l
    for i, frame in enumerate(l):
        if seq == frame.getSequenceNum():
            # node.warn(f"List len {len(l)} Frame with same seq num: {i},seq {seq}")
            l = l[i:]
            break
    return l[0]

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb
while True:
    preview = node.io['preview'].tryGet()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}")
        l.append(preview)

    face_dets = node.io['face_det_in'].tryGet()
    # node.warn(f"Faces detected: {len(face_dets)}")
    if face_dets is not None:
        passthrough = node.io['passthrough'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue
        img = get_latest_frame(seq)

        for i, det in enumerate(face_dets.detections):
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            # node.warn(f"Sending {i + 1}. age/gender det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg.setResize(62, 62)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
""")
    cam.preview.link(image_manip_script.inputs['preview'])

    age_gender_manip = pipeline.create(dai.node.ImageManip)
    age_gender_manip.initialConfig.setResize(62, 62)
    age_gender_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(age_gender_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(age_gender_manip.inputImage)

    # Age/Gender second stange NN
    print("Creating Age Gender Neural Network...")
    age_gender_nn = pipeline.create(dai.node.NeuralNetwork)
    age_gender_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=6))
    age_gender_manip.out.link(age_gender_nn.input)

    age_gender_nn_xout = pipeline.create(dai.node.XLinkOut)
    age_gender_nn_xout.setStreamName("recognition")
    age_gender_nn.out.link(age_gender_nn_xout.input)

    print("Pipeline created.")
    return pipeline

with dai.Device(create_pipeline()) as device:
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    sync = TwoStageHostSeqSync()
    queues = {}
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)

    detections = []
    results = []

    fps = FPS()
    fps.start()

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and age/gender recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            recognitions = msgs["recognition"]

            for i, detection in enumerate(detections):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                rec = recognitions[i]

                age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
                gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
                gender_str = "female" if gender[0] > gender[1] else "male"

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                y = (bbox[1] + bbox[3]) // 2
                cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                # You could also get result["3d"].x and result["3d"].y coordinates
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
