import depthai as dai
from host_wls_filter import WLSFilter
from host_display import Display

LR_CHECK = False  # Better handling for occlusions

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = left.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.NV12)

    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = right.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_5x5)
    stereo.setRectifyEdgeFillColor(
        0
    )  # Black, to better see the cutout from rectification (black stripe on the edges)
    stereo.setLeftRightCheck(LR_CHECK)
    stereo.setSubpixel(False)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)

    wls_filter = pipeline.create(WLSFilter).build(
        disparity=stereo.disparity,
        rectified_right=stereo.rectifiedRight,
        max_disparity=stereo.initialConfig.getMaxDisparity(),
    )

    right_frame = pipeline.create(Display).build(frames=wls_filter.right_frame)
    right_frame.setName("rectified right")

    disparity_frame = pipeline.create(Display).build(frames=wls_filter.disparity_frame)
    disparity_frame.setName("disparity")

    depth_frame = pipeline.create(Display).build(frames=wls_filter.depth_frame)
    depth_frame.setName("wls raw depth")

    filtered_disp = pipeline.create(Display).build(frames=wls_filter.filtered_disp)
    filtered_disp.setName("wlsFilter")

    colored_disp = pipeline.create(Display).build(frames=wls_filter.colored_disp)
    colored_disp.setName("wls colored disp")

    print("Pipeline created.")
    pipeline.run()
