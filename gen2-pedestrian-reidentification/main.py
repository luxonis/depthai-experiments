from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler

def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def create_pipeline(stereo):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1632, 960)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)

    # ImageManip will resize the frame before sending it to the Face detection NN node
    person_det_manip = pipeline.create(dai.node.ImageManip)
    person_det_manip.initialConfig.setResize(544, 320)
    person_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    cam.preview.link(person_det_manip.inputImage)

    person_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    person_nn.setConfidenceThreshold(0.5)
    person_nn.setBlobPath(blobconverter.from_zoo(name="person-detection-retail-0013", shaves=6))
    person_det_manip.out.link(person_nn.input)

    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    person_nn.out.link(face_det_xout.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'recognition_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    cam.preview.link(image_manip_script.inputs['preview'])
    person_nn.out.link(image_manip_script.inputs['nn_in'])
    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    person_nn.passthrough.link(image_manip_script.inputs['passthrough'])

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

        face_dets = node.io['nn_in'].tryGet()
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
                cfg.setResize(128, 256)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)
    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(128, 256)
    recognition_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

    # Age/Gender second stange NN
    print("Creating Age Gender Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)

    recognition_nn.setBlobPath(blobconverter.from_zoo(name="person-reidentification-retail-0288", shaves=6))
    recognition_manip.out.link(recognition_nn.input)

    recognition_nn_xout = pipeline.create(dai.node.XLinkOut)
    recognition_nn_xout.setStreamName("recognition")
    recognition_nn.out.link(recognition_nn_xout.input)

    return pipeline

with dai.Device() as device:
    stereo = 1 < len(device.getConnectedCameras())
    stereo=False
    device.startPipeline(create_pipeline(stereo))

    sync = TwoStageHostSeqSync()
    fps = FPSHandler()
    queues = {}
    results = []
    # Create output queues
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)

    while True:
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and age/gender recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)

        msgs = sync.get_msgs()
        if msgs is not None:
            fps.nextIter()
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            recognitions = msgs["recognition"]

            txt = "FPS: {:.1f}".format(fps.fps())
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

            for i, detection in enumerate(detections):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                reid_result = recognitions[i].getFirstLayerFp16()
                # print('result', reid_result)

                for i, vector in enumerate(results):
                    # print(f"Checking vector {i}")
                    dist = cos_dist(reid_result, vector)
                    if dist > 0.7:
                        results[i] = np.array(reid_result)
                        break
                else:
                    # print("adding new vector")
                    results.append(np.array(reid_result))

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                y = (bbox[1] + bbox[3]) // 2
                cv2.putText(frame, f"Person {i}", (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                cv2.putText(frame, f"Person {i}", (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)

            cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break