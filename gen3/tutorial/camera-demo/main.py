import depthai as dai

from host_display import Display
from host_depth_color_transform import DepthColorTransform

color_resolution = (1280, 720)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    color_output = color.requestOutput(color_resolution, dai.ImgFrame.Type.BGR888i)
    left_output = left.requestFullResolutionOutput()
    right_output = right.requestFullResolutionOutput()

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*color_resolution)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_5x5)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    depth_parser = pipeline.create(DepthColorTransform).build(
        stereo.disparity,
        stereo.initialConfig.getMaxDisparity()
    )

    color_display = pipeline.create(Display).build(color_output)
    color_display.setName("Color camera")
    left_display = pipeline.create(Display).build(left_output)
    left_display.setName("Left camera")
    right_display = pipeline.create(Display).build(right_output)
    right_display.setName("Right camera")
    disparity_display = pipeline.create(Display).build(depth_parser.output)
    disparity_display.setName("Disparity")

    print("Pipeline created.")
    pipeline.run()
