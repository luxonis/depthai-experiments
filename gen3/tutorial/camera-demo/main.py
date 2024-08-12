import depthai as dai
from display_nodes import DisplayMono, DisplayRGB, DisplayStereo


lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False   # Better accuracy for longer distance, fractional disparity 32-levels
median   = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)


def test_rgb_cam_pipeline(pipeline : dai.Pipeline):
    print("Creating rgb cam pipeline")
    cam = pipeline.create(dai.node.ColorCamera)

    cam.setPreviewSize(540, 540)
    cam.setVideoSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    pipeline.create(DisplayRGB).build(
        rgbPreview=cam.preview,
        rgbVideo=cam.video
    )   


def test_stereo_depth_pipeline(pipeline : dai.Pipeline):
    print("Creating Stereo Depth pipeline")

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)

    cam_left .setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    for cam in [cam_left, cam_right]:
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam.setFps(20.0)

    stereo = pipeline.create(dai.node.StereoDepth).build(cam_left.out, cam_right.out)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(median) 
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    pipeline.create(DisplayStereo).build(
        monoLeftOut=cam_left.out,
        monoRightOut=cam_right.out,
        dispOut=stereo.disparity,
    )


def test_mono_cam_pipeline(pipeline : dai.Pipeline):
    print("Creating mono cams pipeline") 

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)

    cam_left .setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam.setFps(20.0)

    pipeline.create(DisplayMono).build(
        monoLeftOut=cam_left.out,
        monoRightOut=cam_right.out
    )


with dai.Pipeline() as pipeline:

    # test_mono_cam_pipeline(pipeline)
    # test_rgb_cam_pipeline(pipeline)
    test_stereo_depth_pipeline(pipeline)

    print("Pipeline created.")
    pipeline.run()
    print("Pipeline finished.")
