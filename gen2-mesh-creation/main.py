import cv2
import numpy as np
import depthai as dai
from projector_3d import PointCloudVisualizer
from pathlib import Path
'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''

curr_path = Path(__file__).parent.resolve()
print("path")
print(curr_path)
Path("./pcl_dataset/depth").mkdir(parents=True, exist_ok=True)
Path("./pcl_dataset/rec_right").mkdir(parents=True, exist_ok=True)
Path("./pcl_dataset/ply").mkdir(parents=True, exist_ok=True)
# StereoDepth config options. TODO move to command line options
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
# right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
right_intrinsic = [[860.858642578125, 0.0,              649.8875732421875], 
                   [0.0,            861.3336791992188, 309.46539306640625], 
                   [0.0,            0.0,                1.0]]

pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
capture_pcl = False



_lambda = 8000
def on_trackbar_change_lambda(value):
    global _lambda
    _lambda = value * 100
    return

_sigma = 1.5
def on_trackbar_change_sigma(value):
    global _sigma
    _sigma = value / float(10)
    return

_lambda_trackbar_name = 'Lambda'
wls_stream = "wls_filte disparityr"
cv2.namedWindow(wls_stream)
_lambda_slider_min = 0
_lambda_slider_max = 255
_lambda_slider_default = 80
cv2.createTrackbar(_lambda_trackbar_name, wls_stream, _lambda_slider_min, _lambda_slider_max, on_trackbar_change_lambda)
cv2.setTrackbarPos(_lambda_trackbar_name, wls_stream, _lambda_slider_default)
wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

_sigma_trackbar_name = 'Sigma'
_sigma_slider_min = 0
_sigma_slider_max = 100
_sigma_slider_default = 15
cv2.createTrackbar(_sigma_trackbar_name, wls_stream, _sigma_slider_min, _sigma_slider_max, on_trackbar_change_sigma)
cv2.setTrackbarPos(_sigma_trackbar_name, wls_stream, _sigma_slider_default)

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
    stereo.setMedianFilter(median) # KERNEL_7x7 default
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
    # stereo.depth         .link(xout_depth.input)
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
    global last_rectif_right, count, capture_pcl
    # global last_rectif_right
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
        wls_filter.setLambda(_lambda)
        wls_filter.setSigmaColor(_sigma)
        filtered_disp = wls_filter.filter(disp, last_rectif_right)

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth_wls = (disp_levels * baseline * focal / filtered_disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)
        
        if 1: # Optionally, extend disparity range to better visualize it
            frame_wls = (filtered_disp * 255. / max_disp).astype(np.uint8)
            cv2.imshow(wls_stream, frame_wls)
        # Optionally, apply a color map
        # frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if 1: # point cloud viewer
            if 0: # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth_wls, frame_rgb, True)
            else: # Option 2: project rectified right
                if capture_pcl:
                    print("capturing...")
                    count += 1
                    depth_path = str(curr_path) + '/pcl_dataset/depth/depth_' + str(count)+'.png'
                    rec_right_path =  str(curr_path) + '/pcl_dataset/rec_right/rec_right' + str(count)+'.png'
                    # print(depth_path)
                    cv2.imwrite(depth_path, depth)
                    cv2.imwrite(rec_right_path, last_rectif_right)
                    capture_pcl = False
                pcl_converter.rgbd_to_projection(depth_wls, last_rectif_right, False)
            pcl_converter.visualize_pcd()

        pcl_converter.visualize_pcd()

    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame

def test_pipeline():
    global capture_pcl
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
            # Skip some streams for now, to reduce CPU load
            if name in ['left', 'right', 'rectified_left', 'depth']: continue
            frame = convert_to_cv2_frame(name, image)
            cv2.imshow(name, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('d'):
            capture_pcl = True
        if key == ord('p'):
            ply_pth = str(curr_path) + '/pcl_dataset/ply/'
            # pcl_converter.save_ply(ply_pth)
            pcl_converter.save_mesh_from_rgbd(ply_pth)
            # pcl_converter.save_mesh_as_ply_vista(ply_pth)

    print("Closing device")
    del device

count = 0
test_pipeline()
