# coding=utf-8
import depthai as dai
import blobconverter

from human_machine_safety import HumanMachineSafety

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

modelDescription = dai.NNModelDescription(modelSlug="mediapipe-palm-detection", platform="RVC2", modelVersionSlug="128x128")
archivePath = dai.getModelFromZoo(modelDescription, useCached=True)

DEPTH_THRESH_HIGH = 3000
DEPTH_THRESH_LOW = 500
WARNING_DIST = 300

# If dangerous object is too close to the palm, warning will be displayed
DANGEROUS_OBJECTS = ["bottle"]

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


device = dai.Device()
with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(2, 3) # To match 720P mono cameras
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.initialControl.setManualFocus(130)
    # For MobileNet NN
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewSize(300, 300)
    cam.setInterleaved(False)
    cam.setFps(15)

    # For Palm-detection NN
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(128, 128)
    cam.preview.link(manip.inputImage)

    print(f"Creating palm detection Neural Network...")
    model_nn = pipeline.create(dai.node.NeuralNetwork)
    manip.out.link(model_nn.input)
    model_nn.setNNArchive(dai.NNArchive(archivePath), 5)
    model_nn.input.setBlocking(False)

    # Creating left/right mono cameras for StereoDepth
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(15)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(15)

    # Create StereoDepth node that will produce the depth map
    stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth).build(left.out, right.out)
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setSubpixelFractionalBits(3)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    sdn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    sdn.setBlob(blobconverter.from_zoo("mobilenet-ssd"))
    sdn.setNumShavesPerInferenceThread(5)
    sdn.setConfidenceThreshold(0.5)
    sdn.input.setBlocking(False)
    sdn.setBoundingBoxScaleFactor(0.2)
    sdn.setDepthLowerThreshold(DEPTH_THRESH_LOW)
    sdn.setDepthUpperThreshold(DEPTH_THRESH_HIGH)

    cam.preview.link(sdn.input)
    stereo.depth.link(sdn.inputDepth)
    
    human_machine_safety = pipeline.create(HumanMachineSafety).build(
        in_rgb=cam.isp,
        in_det=sdn.out,
        in_depth=sdn.passthroughDepth,
        palm_in=model_nn.out,
        label_map=labelMap,
        dangerous_objects=DANGEROUS_OBJECTS
    )
    human_machine_safety.set_depth_thresh_low(DEPTH_THRESH_LOW)
    human_machine_safety.set_depth_thresh_high(DEPTH_THRESH_HIGH)
    human_machine_safety.set_warning_dist(WARNING_DIST)

    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    
    print("Pipeline created.")
    pipeline.run()
