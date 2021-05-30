#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

'''
Blob taken from the great PINTO zoo

git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob
'''

parser = argparse.ArgumentParser()
parser.add_argument("-shape", "--nn_shape", help="select NN model shape", default=256, type=int)
parser.add_argument("-nn", "--nn_path", help="select model path for inference", default='models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob', type=str)
args = parser.parse_args()

nn_shape = args.nn_shape
nn_path = args.nn_path

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.2,0)

def dispay_colored_depth(frame, name):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
    cv2.imshow(name, frame_colored)

class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]

# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(2, 3) # To match 720P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)

# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(nn_shape, nn_shape)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.createXLinkOut()
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Left mono camera
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(245)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Depth output is 1280x720. Color frame has 1:1 (w:h) ratio
# 720/1280=0.5625  1-0.625=0.4375  0.375/2=0.21875  1-0.21875=0.78125
topLeft = dai.Point2f(0.21875, 0)
bottomRight = dai.Point2f(0.78125, 1)
# This ROI will convert 1280x720 depth frame into 720 depth frame
crop_depth = pipeline.createImageManip()
crop_depth.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
stereo.depth.link(crop_depth.inputImage)

# Create depth output
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
crop_depth.out.link(xout_depth.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    frame = None
    depth_frame = None

    while True:
        in_color = q_color.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_color is not None:
            frame = in_color.getCvFrame()
            frame = crop_to_square(frame)
            frame = cv2.resize(frame, (720,720))

        if in_depth is not None:
            depth_frame = in_depth.getFrame()
            # Comment this out to disable showing of depth frame
            dispay_colored_depth(depth_frame, "depth")

        if in_nn is not None:
            fps.next_iter()

            # get layer1 data
            layer1 = in_nn.getFirstLayerInt32()
            # reshape to numpy array
            lay1 = np.asarray(layer1, dtype=np.int32).reshape((nn_shape, nn_shape))
            output_colors = decode_deeplabv3p(lay1)
            print("output_colors1", output_colors.shape)
            # To match 720x720 depth frames
            output_colors = cv2.resize(output_colors, (720, 720))
            print("output_colors2", output_colors.shape)

            if frame is not None:
                frame = show_deeplabv3p(output_colors, frame)
                cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
                cv2.imshow("weighted", frame)

            if depth_frame is not None:
                depth_overlay = depth_frame * output_colors
                cv2.imshow("depth_overlay",depth_overlay)
                # You can add custom code here, for example depth averaging

        if cv2.waitKey(1) == ord('q'):
            break
