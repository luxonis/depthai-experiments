import cv2
import depthai as dai
import blobconverter


MOBILENET_DETECTOR_PATH = str(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
OBJECTRON_CHAIR_PATH = "models/objectron_chair_openvino_2021.4_6shave.blob"
INPUT_W, INPUT_H = 427, 267

with dai.Pipeline() as pipeline:

    # Create a camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setIspScale(1, 3)
    cam.initialControl.setManualFocus(130)
    cam.setPreviewSize(INPUT_W, INPUT_H)
    cam.setInterleaved(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setPreviewKeepAspectRatio(True)
    cam.setFps(30)

    # Create a Manip that resizes image before detection
    manip_det = pipeline.create(dai.node.ImageManip)
    manip_det.initialConfig.setResize(300, 300)
    manip_det.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    
    # Mobilenet parameters
    nn_det = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn_det.setBlobPath(MOBILENET_DETECTOR_PATH)
    nn_det.setConfidenceThreshold(0.5)
    nn_det.setNumInferenceThreads(2)

    # Connect Camera to Manip
    cam.preview.link(manip_det.inputImage)
    manip_det.out.link(nn_det.input)

    # ScriptNode - take the output of the detector, postprocess, and send info to ImageManip for resizing
    # the input to the pose NN
    script_pp = pipeline.create(dai.node.Script)
    nn_det.out.link(script_pp.inputs["det_in"])
    nn_det.passthrough.link(script_pp.inputs['det_passthrough'])

    f = open("script.py", "r")
    script_txt = f.read()
    script_pp.setScript(script_txt)

    script_pp.setProcessor(dai.ProcessorType.LEON_CSS)

    # Connect camera preview to scripts input to use full image size
    cam.preview.link(script_pp.inputs['preview'])

    manip_pose = pipeline.create(dai.node.ImageManip)
    manip_pose.initialConfig.setResize(224, 224)
    manip_pose.inputConfig.setWaitForMessage(True)
    manip_pose.inputConfig.setMaxSize(10)
    manip_pose.inputImage.setMaxSize(10)

    # Link scripts output to pose Manip
    script_pp.outputs['manip_cfg'].link(manip_pose.inputConfig)
    script_pp.outputs['manip_img'].link(manip_pose.inputImage)

    # 2nd stage pose NN
    nn_pose = pipeline.create(dai.node.NeuralNetwork)
    nn_pose.setBlobPath(OBJECTRON_CHAIR_PATH)
    manip_pose.out.link(nn_pose.input)
    nn_pose.input.setMaxSize(20)
    

    camQ = cam.preview.createOutputQueue()
    qDet = nn_det.out.createOutputQueue()
    qCfg = script_pp.outputs['manip_cfg'].createOutputQueue()
    qPose = nn_pose.out.createOutputQueue()
    qPoseFrame = nn_pose.passthrough.createOutputQueue()

    print("pipeline starting")
    pipeline.start()
    print("pipeline started`")

    while pipeline.isRunning():
        
        in_frame : dai.ImgFrame= camQ.tryGet()
        det = qDet.tryGet()
        cfg = qCfg.tryGet()
        pose = qPose.tryGet()
        poseFrame = qPoseFrame.tryGet()

        if in_frame is not None:
            cv2.imshow("preview", in_frame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break

