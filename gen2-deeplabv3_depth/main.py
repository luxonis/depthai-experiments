#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
from datetime import datetime, timedelta

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
TARGET_SHAPE = (400,400)

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def get_multiplier(output_tensor):
    class_binary = [[0], [1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.5,0)

def dispay_colored_depth(frame, name):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
    cv2.imshow(name, frame_colored)
    return frame_colored

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

class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        self.arrays[name].append(msg)
    def get_msgs(self, timestamp):
        ret = {}
        for name, arr in self.arrays.items():
            for i, msg in enumerate(arr):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=20):
                    ret[name] = msg
                    self.arrays[name] = arr[i:]
                    break
        return ret


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
# Color cam: 1920x1080
# Mono cam: 640x400
cam.setIspScale(2,3) # To match 400P mono cameras
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

xout_passthrough = pipeline.createXLinkOut()
xout_passthrough.setStreamName("pass")
# Only send metadata, we are only interested in timestamp, so we can sync
# depth frames with NN output
xout_passthrough.setMetadataOnly(True)
detection_nn.passthrough.link(xout_passthrough.input)

# Left mono camera
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(245)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create depth output
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_pass = device.getOutputQueue(name="pass", maxSize=4, blocking=False)

    fps = FPSHandler()
    sync = HostSync()
    depth_frame = None

    while True:
        sync.add_msg("color", q_color.get())

        in_depth = q_depth.tryGet()
        if in_depth is not None:
            sync.add_msg("depth", in_depth)

        in_nn = q_nn.tryGet()
        if in_nn is not None:
            fps.next_iter()
            # Get NN output timestamp from the passthrough
            timestamp = q_pass.get().getTimestamp()
            msgs = sync.get_msgs(timestamp)

            # get layer1 data
            layer1 = in_nn.getFirstLayerInt32()
            # reshape to numpy array
            lay1 = np.asarray(layer1, dtype=np.int32).reshape((nn_shape, nn_shape))
            output_colors = decode_deeplabv3p(lay1)

            # To match depth frames
            output_colors = cv2.resize(output_colors, TARGET_SHAPE)

            if "color" in msgs:
                frame = msgs["color"].getCvFrame()
                frame = crop_to_square(frame)
                frame = cv2.resize(frame, TARGET_SHAPE)

                frame = show_deeplabv3p(output_colors, frame)
                cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
                cv2.imshow("weighted", frame)

            if "depth" in msgs:
                depth_frame = msgs["depth"].getFrame()
                depth_frame = crop_to_square(depth_frame)
                depth_frame = cv2.resize(depth_frame, TARGET_SHAPE)

                # Optionally display depth frame with deeplab detection
                colored_depth_frame = dispay_colored_depth(depth_frame, "depth")
                colored_depth_frame = show_deeplabv3p(output_colors, colored_depth_frame)
                cv2.imshow("weighted depth", colored_depth_frame)

                multiplier = get_multiplier(lay1)
                multiplier = cv2.resize(multiplier, TARGET_SHAPE)
                depth_overlay = depth_frame * multiplier
                dispay_colored_depth(depth_overlay, "depth_overlay")
                # You can add custom code here, for example depth averaging

        if cv2.waitKey(1) == ord('q'):
            break
