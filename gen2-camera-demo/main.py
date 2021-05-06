#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse

'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - median filtering is disabled on device. TODO enable.
 - with subpixel, if both depth and disparity are used, only depth is valid.
'''

parser = argparse.ArgumentParser()
parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
parser.add_argument("-static", "--static_frames", default=False, action="store_true",
                    help="Run stereo on static frames passed from host 'dataset' folder")
parser.add_argument("-rgb", "--enable_rgb", const="1080", choices={"1080", "4k", "12mp"}, nargs="?",
                    help="Add RGB camera to the pipeline, while also optionally "
                    "selecting resolution (default: %(const)s)")
parser.add_argument("-depth", "--enable_depth", default=False, action="store_true",
                    help="Enable StereoDepth 'depth' output. By default only disparity "
                    "is enabled")
parser.add_argument("-conf", "--enable_confidence", default=False, action="store_true",
                    help="Enable StereoDepth confidence map output")
parser.add_argument("-ns", "--no_stereo", default=False, action="store_true",
                    help="Disable stereo")
parser.add_argument("-lrc", "--lr_check", type=int, default=1, choices={0, 1},
                    help="Enable 'Left-Right Check' mode, providing better "
                    "handling for occlusions. App default: %(default)s")
parser.add_argument("-ext", "--extended_disparity", type=int, default=0, choices={0, 1},
                    help="Enable 'Extended Disparity' mode, for a closer-in minimum "
                    "distance, disparity range is doubled. App default: %(default)s")
parser.add_argument("-sub", "--subpixel", type=int, default=1, choices={0, 1},
                    help="Enable 'Subpixel' mode, for better accuracy at longer "
                    "distance, fractional disparity 32-levels. App default: %(default)s")
parser.add_argument("-med", "--median_size", default=7, type=int, choices={0, 3, 5, 7},
                    help="Disparity / depth median filter kernel size (N x N) . "
                    "0 = filtering disabled. Default: %(default)s")
parser.add_argument("-pw", "--preview_width", type=int, default=300,
                    help="RGB preview width")
parser.add_argument("-ph", "--preview_height", type=int, default=300,
                    help="RGB preview height")
parser.add_argument("-manip", "--preview_manip", default=False, action="store_true",
                    help="Use ImageManip to generate RGB preview")
parser.add_argument("-nn", "--run_nn", const="rgb", choices={"rgb", "left", "right"}, nargs="?",
                    help="Run NN on the selected camera (default: %(const)s)")

stereo_align_options = {
    'left': dai.StereoDepthProperties.DepthAlign.RECTIFIED_LEFT,
    "right": dai.StereoDepthProperties.DepthAlign.RECTIFIED_RIGHT,
    'rgb': dai.CameraBoardSocket.RGB,
    'center': dai.StereoDepthProperties.DepthAlign.CENTER
}

parser.add_argument("-sa", "--stereo_align", const="right", choices=stereo_align_options, nargs="?",
                    help="Set stereo alignment (default: %(const)s)")


args = parser.parse_args()

point_cloud = args.pointcloud  # Create point cloud visualizer. Depends on 'out_rectified'
#print(args.enable_rgb)
enable_rgb = args.enable_rgb
enable_stereo = not args.no_stereo
enable_nn = args.run_nn
stereo_align = stereo_align_options[args.stereo_align] if args.stereo_align is not None else stereo_align_options['right']
if enable_nn == 'rgb': enable_rgb = True

#stereo_align = 

# StereoDepth config options. TODO move to command line options
source_camera = not args.static_frames
out_depth = args.enable_depth
out_confidence = args.enable_confidence
out_rectified = 1  # Output and display rectified streams
lrcheck = args.lr_check
extended = args.extended_disparity
subpixel = args.subpixel

median_opts = {
    0: dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    3: dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    5: dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    7: dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
median = median_opts.get(args.median_size)


rgb_res_opts = {
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k'  : dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
}
rgb_res = rgb_res_opts.get(args.enable_rgb)

last_frame_rgb_video = None

if enable_rgb and stereo_align != dai.CameraBoardSocket.RGB:
    if args.stereo_align is None:
        print('Force enabling --stereo_align rgb because RGB camera enabled.')
        stereo_align = stereo_align_options['rgb']
    else:
        print('Warning: --stereo_align should be set to rgb when -rgb is enabled.')


print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)
print("    Depth output:      ", out_depth)
print("    Stereo Align:      ", stereo_align)
if args.enable_rgb:
    print("    Enable RGB:        ", True)

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


def build_pipeline(pipeline):
    streams = []
    if enable_rgb:
        print("Adding to pipeline: RGB CAM -> XLINK OUT")
        cam = pipeline.createColorCamera()
        xout_preview = pipeline.createXLinkOut()
        xout_video = pipeline.createXLinkOut()
    
        cam.setPreviewSize(args.preview_width, args.preview_height)
        # cam.setPreviewKeepAspectRatio(False)
        cam.setResolution(rgb_res)
        cam.setIspScale(2, 3)
        
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.initialControl.setManualFocus(130)
    
        xout_preview.setStreamName('rgb_preview')
        xout_video  .setStreamName('rgb_video')
    
        streams.append('rgb_video')
        if 1:
            cam.isp.link(xout_video.input)
        else:
            cam.video.link(xout_video.input)
    
        if enable_nn:
            streams.append('rgb_preview')
            cam.preview.link(xout_preview.input)

# TODO ISP out test
    # print("==== ", cam.getIspSize(), cam.getIspWidth(), cam.getIspHeight())

    if enable_stereo:
        print("Adding Stereo Depth to pipeline: ", end='')
        if source_camera:
            print("MONO CAMS -> STEREO -> XLINK OUT")
        else:
            print("XLINK IN -> STEREO -> XLINK OUT")
    
        if source_camera:
            cam_left = pipeline.createMonoCamera()
            cam_right = pipeline.createMonoCamera()
        else:
            cam_left = pipeline.createXLinkIn()
            cam_right = pipeline.createXLinkIn()
        stereo = pipeline.createStereoDepth()
        xout_left = pipeline.createXLinkOut()
        xout_right = pipeline.createXLinkOut()
        xout_depth = pipeline.createXLinkOut()
        xout_disparity = pipeline.createXLinkOut()
        xout_confidence = pipeline.createXLinkOut()
        xout_rectif_left = pipeline.createXLinkOut()
        xout_rectif_right = pipeline.createXLinkOut()
    
        if source_camera:
            cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
            cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            for c in [cam_left, cam_right]:  # Common config
                c.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
                c.setFps(30.0)
        else:
            cam_left .setStreamName('in_left')
            cam_right.setStreamName('in_right')
    
    #    stereo.setOutputDepth(out_depth)
    #    stereo.setOutputRectified(out_rectified)
        #stereo.setBaselineOverrideCm(123.4)
        #stereo.setFovOverrideDegrees(1)
        stereo.setConfidenceThreshold(100)
        stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
        stereo.setMedianFilter(median)  # KERNEL_7x7 default
        stereo.setLeftRightCheck(lrcheck)
        stereo.setExtendedDisparity(extended)
        stereo.setSubpixel(subpixel)
        stereo.setDepthAlign(stereo_align)

        if source_camera:
            # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
            # stereo.loadCalibrationFile(path)
            pass
        else:
            stereo.setEmptyCalibration()  # Set if the input frames are already rectified
            stereo.setInputResolution(1280, 720)
    
        xout_left        .setStreamName('left')
        xout_right       .setStreamName('right')
        xout_depth       .setStreamName('depth')
        xout_disparity   .setStreamName('disparity')
        xout_confidence  .setStreamName('confidence')
        xout_rectif_left .setStreamName('rectified_left')
        xout_rectif_right.setStreamName('rectified_right')

        streams.extend(['left', 'right'])

        cam_left .out        .link(stereo.left)
        cam_right.out        .link(stereo.right)
        stereo.syncedLeft    .link(xout_left.input)
        stereo.syncedRight   .link(xout_right.input)
        if out_rectified:
            stereo.rectifiedLeft .link(xout_rectif_left.input)
            stereo.rectifiedRight.link(xout_rectif_right.input)
            streams.extend(['rectified_left', 'rectified_right'])
        if not (out_depth and subpixel):
            stereo.disparity     .link(xout_disparity.input)
            streams.append('disparity')
        if out_depth:
            stereo.depth.link(xout_depth.input)
            streams.append('depth')
        if out_confidence:
            stereo.confidence.link(xout_confidence.input)
            streams.append('confidence')

    if enable_nn:
        mobilenet_path = '/home/user/Downloads/mobilenet.blob'
        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(mobilenet_path)
        if enable_nn == 'rgb':
            cam.preview.link(detection_nn.input)
        else:
            manip = pipeline.createImageManip()
            manip.setResize(args.preview_width, args.preview_height)
#            manip.setResizeThumbnail(args.preview_width, args.preview_height, 0, 0, 0)

            manip.setMaxOutputFrameSize(1920 * 1080 * 3)
            manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
            xout_manip = pipeline.createXLinkOut()
            xout_manip.setStreamName('manip')
            
            # manip.setKeepAspectRatio(False)
            if enable_nn == 'left':
                stereo.rectifiedLeft.link(manip.inputImage)
            elif enable_nn == 'right':
                stereo.rectifiedRight.link(manip.inputImage)
            manip.out.link(xout_manip.input)
            manip.out.link(detection_nn.input)
            
            streams.append("manip")

        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        detection_nn.out.link(xout_nn.input)

        streams.append('nn')

    return streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    global last_rectif_right
    global last_frame_rgb_video
    baseline = 75  # mm
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
    if name in ['rgb_preview', 'manip']:
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video':  # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_IYUV)
        last_frame_rgb_video = frame
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        frame = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / frame).astype(np.uint16)

        if 1:  # Optionally, extend disparity range to better visualize it
            frame = (frame * 255. / max_disp).astype(np.uint8)

        if 1:  # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if pcl_converter is not None:
            if 0:  # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
            else:  # Option 2: project rectified right or rgb
                if enable_rgb:
                    if last_frame_rgb_video is not None:
                        if len(last_frame_rgb_video) != 720:
                            project_frame = cv2.resize(last_frame_rgb_video, (1280,720))
                        else:
                            project_frame = last_frame_rgb_video
                        project_frame = cv2.cvtColor(project_frame, cv2.COLOR_BGR2RGB)
                        pcl_converter.rgbd_to_projection(depth, project_frame, True)
                else:
                    if last_rectif_right is not None:
                        pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
            
            pcl_converter.visualize_pcd()

    else:  # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame


def test_pipeline():
    pipeline = dai.Pipeline()
    streams = build_pipeline(pipeline)

    global last_rectif_right
    global last_frame_rgb_video

    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
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
            q = device.getOutputQueue(s, 8, blocking=False)
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
                    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720 * 1280)
                    tstamp = datetime.timedelta(seconds=timestamp_ms // 1000,
                                                milliseconds=timestamp_ms % 1000)
                    img = dai.ImgFrame()
                    img.setData(data)
                    img.setTimestamp(tstamp)
                    img.setWidth(1280)
                    img.setHeight(720)
                    q.send(img)
                    print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
                timestamp_ms += frame_interval_ms
                index = (index + 1) % dataset_size
                if 1:  # Optional delay between iterations, host driven pipeline
                    sleep(frame_interval_ms / 1000)
            # Handle output streams
            for q in q_list:
                name = q.getName()
                if name == 'nn':
                    in_nn = q.tryGet()
                    if in_nn is not None:
                        print("Got NN output")
                    continue
                image = q.tryGet()
                if image is None:
                    continue
                #print("Received frame:", name, "tstamp", image.getTimestamp().total_seconds())
                # Skip some streams for now, to reduce CPU load
                if name in ['left', 'right']: continue
                frame = convert_to_cv2_frame(name, image)
                cv2.imshow(name, frame)
            if cv2.waitKey(1) == ord('q'):
                break


test_pipeline()
