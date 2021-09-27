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
out_depth     = False  # Disparity by default
out_rectified = True   # Output and display rectified streams
lrcheck       = True #   # Better handling for occlusions
extended      = False  # Closer-in minimum depth, disparity range is doubled
subpixel      = True   # Better accuracy for longer distance, fractional disparity 32-levels
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

def test_pipeline():
    global capture_pcl
    pipeline = dai.Pipeline()

    cam_left          = pipeline.createMonoCamera()
    cam_right         = pipeline.createMonoCamera()
    cam_rgb           = pipeline.createColorCamera()

    stereo            = pipeline.createStereoDepth()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rgb_isp      = pipeline.createXLinkOut()

    cam_left .setCamId(1)
    cam_right.setCamId(2)
    cam_rgb.setCamId(0)

    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setCamId(0)
    cam_rgb.setIspScale(2, 6)

    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
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
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rgb_isp     .setStreamName("rgb")

    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    # stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    cam_rgb.isp.link(xout_rgb_isp.input)
    
    streams = ['disparity', 'rgb']

    print("Creating DepthAI device")
    device = dai.Device()
    print("Starting pipeline")
    calibData = device.readCalibration()
    device.startPipeline(pipeline)

    rgb_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, int(1280/2), int(720/2)))
    print("rgb_intrinsic {}".format(rgb_intrinsic))
    pcl_converter = PointCloudVisualizer(rgb_intrinsic, int(1280/2), int(720/2))


    # Create a receive queue for each stream
    q_list = []
    for s in streams:
        q = device.getOutputQueue(s, 8, True)
        q_list.append(q)
    recent_rgb = None
    recent_disp = None
    max_disp = 96
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32
    baseline = 75 #mm
    focal = rgb_intrinsic[0][0]

    while True:
        for q in q_list:
            name  = q.getName()
            image = q.getAll()[-1]
            if name == 'rgb':
                recent_rgb = image
                frame = image.getCvFrame()
                recent_rgb = frame
                cv2.imshow(name, frame)
            else:
                recent_disp = image.getCvFrame()
                frame = (recent_disp * (255 / stereo.getMaxDisparity())).astype(np.uint8)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                cv2.imshow("disparity_color", frame)

                if recent_rgb is not None:
                    # disp = np.array(frame).astype(np.uint8).view(np.uint16).reshape((h, w))
                   
                    wls_filter.setLambda(_lambda)
                    wls_filter.setSigmaColor(_sigma)
                    filtered_disp = wls_filter.filter(recent_disp, recent_rgb)
                    with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
                        depth_wls = (disp_levels * baseline * focal / recent_disp).astype(np.uint16)
                        # cv2.imshow("wls_stream", depth_wls)
                        recent_rgb_ordered = cv2.cvtColor(recent_rgb, cv2.COLOR_BGR2RGB)
                        pcl_converter.rgbd_to_projection(depth_wls, recent_rgb_ordered, True)
                        pcl_converter.visualize_pcd()
                    if 1: # Optionally, extend disparity range to better visualize it
                        frame_wls = (filtered_disp * 255. / max_disp).astype(np.uint8)
                        frame_wls_color_map = cv2.applyColorMap(frame_wls, cv2.COLORMAP_JET)
                        wighted = cv2.addWeighted(recent_rgb, 0.6, frame, 0.4, 0)
                        cv2.imshow("rgb-disparity", wighted)
                        cv2.imshow(wls_stream, frame_wls)
                        
                        #print("Received frame:", name)
            # Skip some streams for now, to reduce CPU load
            
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('d'):
            capture_pcl = True
        if key == ord('p'):
            print('Capturing.....')
            ply_pth = str(curr_path) + '/pcl_dataset/ply/'
            # pcl_converter.save_ply(ply_pth)
            pcl_converter.save_mesh_from_rgbd(ply_pth)
            # pcl_converter.save_mesh_as_ply_vista(ply_pth)

    print("Closing device")
    del device

count = 0
test_pipeline()
