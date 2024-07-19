import cv2
import depthai as dai
import blobconverter


MOBILENET_DETECTOR_PATH = str(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
OBJECTRON_CHAIR_PATH = "models/objectron_chair_openvino_2021.4_6shave.blob"
INPUT_W, INPUT_H = 427, 267

def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # Create a camera
    cam = pipeline.createColorCamera()
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
    manip_det.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)

    # Mobilenet parameters
    nn_det = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn_det.setBlobPath(MOBILENET_DETECTOR_PATH)
    nn_det.setConfidenceThreshold(0.5)
    nn_det.setNumInferenceThreads(2)

    # Connect Camera to Manip
    cam.preview.link(manip_det.inputImage)
    manip_det.out.link(nn_det.input)

    # Out links
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam")
    cam.preview.link(cam_xout.input)

    nn_det_xout = pipeline.createXLinkOut()
    nn_det_xout.setStreamName("nn_det")
    nn_det.out.link(nn_det_xout.input)

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

    # Create Manip for frame resizing before pose NN
    manip_pose = pipeline.createImageManip()
    manip_pose.initialConfig.setResize(224, 224)
    manip_pose.inputConfig.setWaitForMessage(True)
    manip_pose.inputConfig.setQueueSize(10)
    manip_pose.inputImage.setQueueSize(10)

    # Link scripts output to pose Manip
    script_pp.outputs['manip_cfg'].link(manip_pose.inputConfig)
    script_pp.outputs['manip_img'].link(manip_pose.inputImage)

    # Link manip config to queue
    config_xout = pipeline.createXLinkOut()
    config_xout.setStreamName("cfg_out")
    script_pp.outputs['manip_cfg'].link(config_xout.input)
    
    # 2nd stage pose NN
    nn_pose = pipeline.createNeuralNetwork()
    nn_pose.setBlobPath(OBJECTRON_CHAIR_PATH)
    manip_pose.out.link(nn_pose.input)
    nn_pose.input.setQueueSize(20)

    nn_pose_xout = pipeline.createXLinkOut()
    nn_pose_xout.setStreamName("nn_pose")
    nn_pose.out.link(nn_pose_xout.input)

    nn_pose_frame_xout = pipeline.createXLinkOut()
    nn_pose_frame_xout.setStreamName("nn_pose_frame")
    nn_pose.passthrough.link(nn_pose_frame_xout.input)

    return pipeline



print("Starting pipeline...")
pipeline = create_pipeline()
print("pipeline created")

with dai.Device(pipeline) as device:

    q_cam = device.getOutputQueue("cam", 4, False)
    q_detection = device.getOutputQueue("nn_det", 4, False)
    q_cfg = device.getOutputQueue("cfg_out", 16, False)
    q_pose = device.getOutputQueue("nn_pose", 20, False)
    q_pose_frame = device.getOutputQueue("nn_pose_frame", 16, False)

    while True:

        in_frame : dai.ImgFrame = q_cam.tryGet()
        in_det = q_detection.tryGet()
        in_cfg = q_cfg.tryGet()
        in_pose = q_pose.tryGet()
        in_pose_frame = q_pose_frame.tryGet()

        if in_frame is None:
            continue

        cv2.imshow("preview", in_frame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
