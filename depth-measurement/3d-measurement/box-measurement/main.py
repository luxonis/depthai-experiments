import time
import depthai as dai
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork

from utils.annotation_node import AnnotationNode
from utils.sync_node import SyncNode


NN_WIDTH, NN_HEIGHT = 512, 320
INPUT_SHAPE = (NN_WIDTH, NN_HEIGHT)

IMG_WIDTH, IMG_HEIGHT = 640, 400
CAMERA_RESOLUTION = (IMG_WIDTH, IMG_HEIGHT)

#device = dai.Device(dai.DeviceInfo('10.11.1.175'))
device = dai.Device()
device.setIrLaserDotProjectorIntensity(1.0)
device.setIrFloodLightIntensity(1)

platform = device.getPlatform()

model_version_slug = "512x320"

model_description = dai.NNModelDescription(
    model="luxonis/yolov8-instance-segmentation-nano-carton:512x320:1.0.0",
    platform=platform.name
)

# Download or retrieve the model from the zoo
archivePath = dai.getModelFromZoo(
    model_description,
    apiKey='tapi.oUtZWL1Ib53fQxESjkwiaw.6Aa5gzkKcmhyIRpWTNtMW20Nw5cHiiqVt-BLYgi00ajRB8jy7e72VFpOehhyNR1gkt2Mn9aUtkSrOBShPuFItw'
)

nn_archive = dai.NNArchive(archivePath)

# Box measurement errors (For testing accuracy)
GROUND_TRUTH_DIMENSIONS = (24.5, 20.0, 15.5)            # Replace with real box dimensions 
total_mae = [0, 0, 0]
total_relative_error = [0, 0, 0]

def read_intrinsics():
    calibData = device.readCalibration2()
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_WIDTH, NN_HEIGHT))  # Because the displayed image is with NN input res 
    fx = M2[0, 0]
    fy = M2[1, 1]
    cx = M2[0, 2]
    cy = M2[1, 2]
    return fx, fy, cx, cy

# Create pipeline

with dai.Pipeline(device) as p:
    # Profiling
    fps = 5

    color = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_output = color.requestOutput(
        CAMERA_RESOLUTION, dai.ImgFrame.Type.RGB888i, fps=fps
    )

    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = p.create(dai.node.StereoDepth).build(
        left=left.requestOutput(CAMERA_RESOLUTION, fps=fps),
        right=right.requestOutput(CAMERA_RESOLUTION, fps=fps),
    )
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.enableDistortionCorrection(True)
    # stereo.setExtendedDisparity(True)
    # stereo.setLeftRightCheck(True)

    align = p.create(dai.node.ImageAlign)
    stereo.depth.link(align.input)
    color_output.link(align.inputAlignTo)

    # For PCL
    rgbd = p.create(dai.node.RGBD).build()
    align.outputAligned.link(rgbd.inDepth)
    color_output.link(rgbd.inColor)

    # For NN 
    manip = p.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(*nn_archive.getInputSize())
    manip.initialConfig.setFrameType(
        dai.ImgFrame.Type.BGR888p if platform == dai.Platform.RVC2 else dai.ImgFrame.Type.BGR888i
    )
    manip.setMaxOutputFrameSize(
        nn_archive.getInputSize()[0] * nn_archive.getInputSize()[1] * 3
    )

    color_output.link(manip.inputImage)

    nn = p.create(ParsingNeuralNetwork).build(
        nn_source=nn_archive, input=manip.out
    )

    if platform == dai.Platform.RVC2:
        nn.setNNArchive(
            nn_archive, numShaves=7
        )

    nn._parsers[0].setConfidenceThreshold(0.7)
    nn._parsers[0].setIouThreshold(0.5)
    nn._parsers[0].setMaskConfidence(0.5)

    sync_node = p.create(SyncNode)
    rgbd.pcl.link(sync_node.inputPCL)
    nn.passthrough.link(sync_node.inputRGB)
    nn.out.link(sync_node.inputDet)

    Annotations = AnnotationNode()
    Annotations.intrinsics = read_intrinsics()
    sync_node.out.link(Annotations.input)

    outputToVisualize = color.requestOutput(
        (640, 400),
        type=dai.ImgFrame.Type.RGB888p,
        fps=fps,
    )

    vis = dai.RemoteConnection(httpPort=8082)

    vis.addTopic("Raw video",  outputToVisualize, "images")
    #vis.addTopic("Video", Annotations.outputRGB, "images")
    vis.addTopic("AnnotationsYOLO", Annotations.outputANN, "img_annotations")
    vis.addTopic("AnnotationsCuboidFit", Annotations.outputANNCuboid, "img_annotations")
    #vis.addTopic("Detections", Annotations.outputPCL, "pointcloud")

    p.start()
    vis.registerPipeline(p)
    while p.isRunning():
        time.sleep(0.1)
