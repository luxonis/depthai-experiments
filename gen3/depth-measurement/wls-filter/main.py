import depthai as dai
from host_wls_filter import WLSFilter

LR_CHECK = False   # Better handling for occlusions

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
    stereo.setLeftRightCheck(LR_CHECK)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    wls_filter = pipeline.create(WLSFilter).build(
        disparity=stereo.disparity,
        rectified_right=stereo.rectifiedRight,
        max_disparity=stereo.initialConfig.getMaxDisparity()
    )

    print("Pipeline created.")
    pipeline.run()
