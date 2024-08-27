import depthai as dai

from host_display import Display
from host_display_depth import DisplayDepth

resolution = (1280, 720)

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_output = left.requestOutput(resolution)
    right_output = right.requestOutput(resolution)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*resolution)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    depth_parser = pipeline.create(DisplayDepth).build(
        stereo.disparity,
        stereo.initialConfig.getMaxDisparity()
    )

    color_display = pipeline.create(Display).build(color.requestOutput(resolution))
    color_display.setName("Color camera")
    left_display = pipeline.create(Display).build(left_output)
    left_display.setName("Left camera")
    right_display = pipeline.create(Display).build(right_output)
    right_display.setName("Right camera")
    disparity_display = pipeline.create(Display).build(depth_parser.output)
    disparity_display.setName("Disparity")

    print("Pipeline created.")
    pipeline.run()
