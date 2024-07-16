#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import os
import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
parser.add_argument("-static", "--static_frames", default=False, action="store_true",
                    help="Run stereo on static frames passed from host 'dataset' folder")
args = parser.parse_args()

point_cloud    = args.pointcloud   # Create point cloud visualizer. Depends on 'out_rectified'

# StereoDepth config options. TODO move to command line options
source_camera  = not args.static_frames
out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthConfig.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthConfig.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
dataset_size = 2

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


def create_video_from_images(nameOfImage, videoName):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'dataset', '0', nameOfImage + '.png')
    videoName = os.path.join(dirname, videoName)

    print(os.path.exists(filename))

    frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(videoName, fourcc, 1.0, (width, height))

    if not video.isOpened():
        raise ValueError("Failed to open VideoWriter.")

    for index in range(0, dataset_size):
        # Read the first image to get the frame dimensions
        filename = os.path.join(dirname, 'dataset', str(index), nameOfImage + '.png')

        frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        height, width = frame.shape

        video.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    # Release the video writer
    video.release()


def create_rgb_cam_pipeline(pipeline : dai.Pipeline):
    print("Creating pipeline: COLOR CAM")
    cam = pipeline.create(dai.node.ColorCamera)

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    pipeline.create(DisplayRGB).build(
        rgbPreview=cam.preview,
        rgbVideo=cam.video
    )   


def create_mono_cam_pipeline(pipeline : dai.Pipeline):
    print("Creating pipeline: MONO CAMS")

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)

    cam_left .setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    pipeline.create(DisplayMono).build(
        monoLeftOut=cam_left.out,
        monoRightOut=cam_right.out
    )


def create_stereo_depth_pipeline(pipeline : dai.Pipeline):
    print("Creating Stereo Depth pipeline: ", end='')
    print("MONO CAMS -> STEREO")

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_right = pipeline.create(dai.node.MonoCamera)

    stereo = pipeline.create(dai.node.StereoDepth).build(cam_left.out, cam_right.out)

    cam_left .setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    pipeline.create(DisplayStereo).build(
        monoLeftOut=cam_left.out,
        monoRightOut=cam_right.out,
        dispOut=stereo.disparity,
        depthOut=stereo.depth
    )


def create_stereo_depth_pipeline_from_dataset(pipeline : dai.Pipeline, videos : list):
    print("video input -> STEREO -> XLINK OUT")

    dirname = os.path.dirname(__file__)

    in_left_video : dai.node.ReplayVideo = pipeline.create(dai.node.ReplayVideo)
    in_left_video.setReplayVideoFile(os.path.join(dirname, videos[0]))
    in_left_video.setOutFrameType(dai.ImgFrame.Type.NV12)
    in_left_video.setLoop(True)

    in_right_video : dai.node.ReplayVideo = pipeline.create(dai.node.ReplayVideo)
    in_right_video.setReplayVideoFile(os.path.join(dirname, videos[1]))
    in_right_video.setOutFrameType(dai.ImgFrame.Type.NV12)
    in_right_video.setLoop(True)

    imageManipLeft = pipeline.create(dai.node.ImageManip)
    imageManipLeft.initialConfig.setResize(1280, 720)
    imageManipLeft.initialConfig.setFrameType(dai.ImgFrame.Type.RAW8)
    in_left_video.out.link(imageManipLeft.inputImage)

    imageManipRight = pipeline.create(dai.node.ImageManip)
    imageManipRight.initialConfig.setResize(1280, 720)
    imageManipRight.initialConfig.setFrameType(dai.ImgFrame.Type.RAW8)
    in_right_video.out.link(imageManipRight.inputImage)

    stereo = pipeline.create(dai.node.StereoDepth).build(imageManipLeft.out, imageManipRight.out)

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    # stereo.setEmptyCalibration() # Set if the input frames are already rectified
    stereo.setInputResolution(1280, 720)


    pipeline.create(DisplayStereo).build(
        monoLeftOut=imageManipLeft.out,
        monoRightOut=imageManipRight.out,
        dispOut=stereo.disparity,
        depthOut=stereo.depth
    )



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
        max_disp *= 32
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
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame


class DisplayMono(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output) -> "DisplayMono":
        self.link_args(monoLeftOut, monoRightOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame) -> None:
        cv2.imshow("left", self.mono_convert_to_cv2_frame(monoLeftFrame))
        cv2.imshow("right", self.mono_convert_to_cv2_frame(monoRightFrame))
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def mono_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        return frame


class DisplayRGB(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def build(self, rgbPreview : dai.Node.Output, rgbVideo : dai.Node.Output) -> "DisplayRGB":
        self.link_args(rgbPreview, rgbVideo)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, rgbPreviewFrame : dai.ImgFrame, rgbVideoFrame : dai.ImgFrame) -> None:
        cv2.imshow("rgb_preview", self.preview_convert_to_cv2_frame(rgbPreviewFrame))
        cv2.imshow("rgb_video", self.video_convert_to_cv2_frame(rgbVideoFrame))
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
    
    def preview_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
        return frame
    
    def video_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return frame


class DisplayStereo(dai.node.HostNode):
    def __init__(self):
        self.baseline = 75 #mm
        self.focal = right_intrinsic[0][0]
        self.max_disp = 96
        self.disp_type = np.uint8
        self.disp_levels = 1
        if (extended):
            self.max_disp *= 2
        if (subpixel):
            self.max_disp *= 32
            self.disp_type = np.uint16  # 5 bits fractional disparity
            self.disp_levels = 32
        super().__init__()

    def build(self, monoLeftOut : dai.Node.Output, monoRightOut : dai.Node.Output, dispOut : dai.Node.Output, depthOut : dai.Node.Output) -> "DisplayStereo":
        self.link_args(monoLeftOut, monoRightOut, dispOut, depthOut)
        self.sendProcessingToPipeline(True)
        return self
    
    def process(self, monoLeftFrame : dai.ImgFrame, monoRightFrame : dai.ImgFrame, dispFrame : dai.ImgFrame, depthFrame : dai.ImgFrame) -> None:
        cv2.imshow("rectified_left", self.mono_convert_to_cv2_frame(monoLeftFrame))
        cv2.imshow("rectified_right", self.mono_convert_to_cv2_frame(monoRightFrame))
        cv2.imshow("disparity", self.disparity_convert_to_cv2_frame(dispFrame))
        # Skip some streams for now, to reduce CPU load
        # cv2.imshow("depth", self.depth_convert_to_cv2_frame(depthFrame))
        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()

    def mono_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        return frame
    
    def depth_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
        return frame
    
    def disparity_convert_to_cv2_frame(self, frame : dai.ImgFrame) -> object:
        data, w, h = frame.getData(), frame.getWidth(), frame.getHeight()
        disp = np.array(data).astype(np.uint8).view(self.disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (self.disp_levels * self.baseline * self.focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / self.max_disp).astype(np.uint8)

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

        return frame




def test_pipeline():
    device = dai.Device()
    with dai.Pipeline(device) as pipeline:

        # create_stereo_depth_pipeline(pipeline)
        # create_rgb_cam_pipeline(pipeline)
        # create_mono_cam_pipeline(pipeline)

        create_video_from_images("in_left", "in_left.mp4")
        create_video_from_images("in_right", "in_right.mp4")

        create_stereo_depth_pipeline_from_dataset(pipeline, ["in_left.mp4", "in_right.mp4"])

 
        pipeline.run()


    # with dai.Device() as device:
    #     # Need to set a timestamp for input frames, for the sync stage in Stereo node
    #     timestamp_ms = 0
    #     index = 0
    #     while True:
    #         # Handle input streams, if any
    #         if in_q_list:
    #             dataset_size = 2  # Number of image pairs
    #             frame_interval_ms = 33
    #             for i, q in enumerate(in_q_list):
    #                 name = q.getName()
    #                 path = 'dataset/' + str(index) + '/' + name + '.png'
    #                 data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
    #                 tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
    #                                             milliseconds = timestamp_ms % 1000)
    #                 img = dai.ImgFrame()
    #                 img.setData(data)
    #                 img.setTimestamp(tstamp)
    #                 img.setInstanceNum(inStreamsCameraID[i])
    #                 img.setType(dai.ImgFrame.Type.RAW8)
    #                 img.setWidth(1280)
    #                 img.setHeight(720)
    #                 q.send(img)
    #                 if timestamp_ms == 0:  # Send twice for first iteration
    #                     q.send(img)
    #                 print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
    #             timestamp_ms += frame_interval_ms
    #             index = (index + 1) % dataset_size
    #             if 1: # Optional delay between iterations, host driven pipeline
    #                 sleep(frame_interval_ms / 1000)


test_pipeline()
