import depthai as dai
import blobconverter
from host_depth_segmentation import DepthSegmentation

nn_shape = (256, 256)
TARGET_SHAPE = (400, 400)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # Color cam: 1920x1080
    # Mono cam: 640x400
    cam.setIspScale(2, 3)  # To match 400P mono cameras
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewSize(nn_shape)
    cam.setInterleaved(False)

    manip_cam = pipeline.create(dai.node.ImageManip)
    manip_cam.initialConfig.setResize(*TARGET_SHAPE)
    cam.isp.link(manip_cam.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blobconverter.from_zoo(name="deeplab_v3_mnv2_256x256", zoo_type="depthai", shaves=6))
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    cam.preview.link(nn.input)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    manip_stereo = pipeline.create(dai.node.ImageManip)
    manip_stereo.initialConfig.setResize(*TARGET_SHAPE)
    stereo.disparity.link(manip_stereo.inputImage)

    depth_segmentation = pipeline.create(DepthSegmentation).build(
        preview=manip_cam.out,
        nn=nn.out,
        disparity=manip_stereo.out,
        nn_shape=nn_shape,
        max_disparity=stereo.initialConfig.getMaxDisparity()
    )

    print("Pipeline created.")
    pipeline.run()
