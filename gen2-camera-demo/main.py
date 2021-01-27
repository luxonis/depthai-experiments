import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse

'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''


parser = argparse.ArgumentParser()
parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
args = parser.parse_args()

point_cloud    = args.pointcloud   # Create point cloud visualizer. Depends on 'out_rectified'

# StereoDepth config options. TODO move to command line options
source_camera  = True   # If False, will read input frames from 'dataset' folder
out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled 
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]

pcl_converter = None
if point_cloud:
    if out_rectified:
        try:
            from projector_3d import PointCloudVisualizer
        except ImportError as e:
            raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
        pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
    else:
        print("Disabling point-cloud visualizer, as out_rectified is not set")

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_mono_cam_pipeline():
    print("Creating pipeline: MONO CAMS -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    xout_left  = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()

    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    xout_left .setStreamName('left')
    xout_right.setStreamName('right')

    cam_left .out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    streams = ['left', 'right']

    return pipeline, streams

def create_stereo_depth_pipeline(from_camera=True):
    print("Creating Stereo Depth pipeline: ", end='')
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()

    if from_camera:
        cam_left      = pipeline.createMonoCamera()
        cam_right     = pipeline.createMonoCamera()
    else:
        cam_left      = pipeline.createXLinkIn()
        cam_right     = pipeline.createXLinkIn()
    stereo            = pipeline.createStereoDepth()
    xout_left         = pipeline.createXLinkOut()
    xout_right        = pipeline.createXLinkOut()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rectif_left  = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()

    if from_camera:
        cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]: # Common config
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            #cam.setFps(20.0)
    else:
        cam_left .setStreamName('in_left')
        cam_right.setStreamName('in_right')

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
        #stereo.loadCalibrationFile(path)
        pass
    else:
        stereo.setEmptyCalibration() # Set if the input frames are already rectified
        stereo.setInputResolution(1280, 720)

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

    streams = ['left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

    return pipeline, streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    global last_rectif_right
    baseline = 75 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

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
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if pcl_converter is not None:
            if 0: # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
            else: # Option 2: project rectified right
                pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
            pcl_converter.visualize_pcd()

    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame

def test_pipeline():
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline, streams = create_stereo_depth_pipeline(source_camera)

    print("Creating DepthAI device")
    if 1:
        device = dai.Device(pipeline)
    else: # For debug mode, with the firmware already loaded
        found, device_info = dai.XLinkConnection.getFirstDevice(
                dai.XLinkDeviceState.X_LINK_BOOTED)
        if found:
            device = dai.Device(pipeline, device_info)
        else:
            raise RuntimeError("Device not found")
    print("Starting pipeline")
    device.startPipeline()

    in_streams = []
    if not source_camera:
        # Reversed order trick:
        # The sync stage on device side has a timeout between receiving left
        # and right frames. In case a delay would occur on host between sending
        # left and right, the timeout will get triggered.
        # We make sure to send first the right frame, then left.
        in_streams.extend(['in_right', 'in_left'])
    in_q_list = []
    for s in in_streams:
        q = device.getInputQueue(s)
        in_q_list.append(q)

    # Create a receive queue for each stream
    q_list = []
    for s in streams:
        q = device.getOutputQueue(s, 8, True)
        q_list.append(q)

    # Need to set a timestamp for input frames, for the sync stage in Stereo node
    timestamp_ms = 0
    index = 0
    while True:
        # Handle input streams, if any
        if in_q_list:
            dataset_size = 2  # Number of image pairs
            frame_interval_ms = 33
            for q in in_q_list:
                name = q.getName()
                path = 'dataset/' + str(index) + '/' + name + '.png'
                data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
                tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                            milliseconds = timestamp_ms % 1000)
                img = dai.ImgFrame()
                img.setData(data)
                img.setTimestamp(tstamp)
                img.setWidth(1280)
                img.setHeight(720)
                q.send(img)
                print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
            timestamp_ms += frame_interval_ms
            index = (index + 1) % dataset_size
            if 1: # Optional delay between iterations, host driven pipeline
                sleep(frame_interval_ms / 1000)
        # Handle output streams
        for q in q_list:
            name  = q.getName()
            image = q.get()
            #print("Received frame:", name)
            # Skip some streams for now, to reduce CPU load
            if name in ['left', 'right', 'depth']: continue
            frame = convert_to_cv2_frame(name, image)
            cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print("Closing device")
    del device

test_pipeline()
