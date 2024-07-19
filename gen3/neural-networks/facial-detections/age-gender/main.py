import blobconverter
import depthai as dai
from host_age_gender import AgeGender
from detections_recognitions_sync import DetectionsRecognitionsSync

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(1080, 1080)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setPreviewNumFramesPool(10)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    face_manip = pipeline.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
    cam.preview.link(face_manip.inputImage)

    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork).build()
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=5))
    face_det_nn.setBoundingBoxScaleFactor(0.8)
    face_det_nn.setDepthLowerThreshold(100)
    face_det_nn.setDepthUpperThreshold(5000)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.input.setMaxSize(1)
    face_manip.out.link(face_det_nn.input)
    stereo.depth.link(face_det_nn.inputDepth)

    script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(script.inputs['face_det_in'])
    cam.preview.link(script.inputs['preview'])
    script.setScript("""
msgs = dict()

def add_msg(msg, name, seq = None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    # node.warn(f"New msg {name}, seq {seq}")

    # Each seq number has it's own dict of msgs
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair

        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2: # 1 frame, 1 detection
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs # Returned synced msgs
    return None

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb

while True:
    preview = node.io['preview'].tryGet()
    if preview is not None:
        add_msg(preview, 'preview')

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        seq = face_dets.getSequenceNum()
        add_msg(face_dets, 'dets', seq)

    sync_msgs = get_msgs()
    if sync_msgs is not None:
        img = sync_msgs['preview']
        dets = sync_msgs['dets']
        for i, det in enumerate(dets.detections):
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(62, 62)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
        """)

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(62, 62)
    recognition_manip.inputConfig.setWaitForMessage(True)
    script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    script.outputs['manip_img'].link(recognition_manip.inputImage)

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=5))
    recognition_manip.out.link(recognition_nn.input)

    sync = pipeline.create(DetectionsRecognitionsSync).build()
    face_det_nn.out.link(sync.input_detections)
    recognition_nn.out.link(sync.input_recognitions)

    age_gender = pipeline.create(AgeGender).build(
        preview=cam.preview,
        detections_recognitions=sync.output
    )

    print("Pipeline created.")
    pipeline.run()
