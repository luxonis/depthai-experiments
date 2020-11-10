import cv2
import numpy as np
import depthai as dai

# StereoDepth config options. TODO move to command line options
out_depth      = False  # Disparity by default
out_rectified  = False  # Output and display rectified streams
lrcheck  = False  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels

# Sanitize some incompatible options
if extended or subpixel:
    out_rectified = False # TODO
'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setCamId(0)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_mono_cam_pipeline():
    print("Creating pipeline: MONO CAMS -> XLINK")
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    xout_left  = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()

    cam_left .setCamId(1)
    cam_right.setCamId(2)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    xout_left .setStreamName('left')
    xout_right.setStreamName('right')

    cam_left .out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    streams = ['left', 'right']

    return pipeline, streams

def create_stereo_depth_pipeline():
    print("Creating Stereo Depth pipeline: MONO CAMS -> STEREO -> XLINK")
    pipeline = dai.Pipeline()

    cam_left          = pipeline.createMonoCamera()
    cam_right         = pipeline.createMonoCamera()
    stereo            = pipeline.createStereoDepth()
    xout_left         = pipeline.createXLinkOut()
    xout_right        = pipeline.createXLinkOut()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rectif_left  = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()

    cam_left .setCamId(1)
    cam_right.setCamId(2)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    #stereo.loadCalibrationFile(path) # Default: EEPROM calib is used
    #stereo->setInputResolution(1280, 720); # Default: resolution is taken from Mono nodes
    #stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    xout_left        .setStreamName('left')
    xout_right       .setStreamName('right')
    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rectif_left .setStreamName('rectified_left')
    xout_rectif_right.setStreamName('rectified_right')

    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    stereo.syncedLeft    .link(xout_left.input)
    stereo.syncedRight   .link(xout_right.input)
    stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    stereo.rectifiedLeft .link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ['left', 'right', 'disparity', 'depth']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])

    return pipeline, streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video': # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        max_disp = 96
        type = np.uint8
        if (extended): max_disp *= 2
        if (subpixel): max_disp *= 32; type = np.uint16  # 5 bits fractional disparity
        disp = np.array(data).astype(np.uint8).view(type).reshape((h, w))
        # Optionally, extend disparity range to better visualize it
        frame = (disp * 255. / max_disp).astype(np.uint8)
        # Optionally, apply a color map
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
    return frame

def test_pipeline():
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline, streams = create_stereo_depth_pipeline()

    print("Creating DepthAI device")
    device = dai.Device()
    print("Starting pipeline")
    device.startPipeline(pipeline)

    # Create a receive queue for each stream
    q_list = []
    for s in streams:
        q = device.getOutputQueue(s, 8, True)
        q_list.append(q)

    while True:
        for q in q_list:
            name  = q.getName()
            image = q.get()
            #print("Received frame:", name)
            frame = convert_to_cv2_frame(name, image)
            cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print("Closing device")
    del device

test_pipeline()
