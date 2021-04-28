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


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

# Right mono camera
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# Left mono camera
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)
stereo.setOutputDepth(True)
stereo.setOutputRectified(True)
stereo.setRectifyMirrorFrame(False)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Right rectified -> NN input to have the most accurate NN output/stereo mapping
manip = pipeline.createImageManip()
manip.setResize(nn_shape,nn_shape)
manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
stereo.rectifiedRight.link(manip.inputImage)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
manip.out.link(detection_nn.input)

# Create depth output
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Create output for right frames
xout_right_rectified = pipeline.createXLinkOut()
xout_right_rectified.setStreamName("right_rectified")
detection_nn.passthrough.link(xout_right_rectified.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queues will be used to get the outputs from the device
    q_right = device.getOutputQueue(name="right_rectified", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0

    frame = None
    depth_frame = None

    while True:
        in_right = q_right.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_right is not None:
            frame = in_right.getCvFrame()

        if in_depth is not None:
            depth_frame = in_depth.getFrame()
            # Crop the depth map
            depth_frame = depth_frame[0:400, 120:520]
            # Resize it so it will match the NN output
            depth_frame = cv2.resize(depth_frame, (nn_shape,nn_shape))

            # Since we are using setRectifyMirrorFrame(False), we have to flip depth frame
            depth_frame = cv2.flip(depth_frame, 1)

            # Remove this to disable showing depth frames
            depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            cv2.imshow("depth", depthFrameColor)

        if in_nn is not None:
            # print("NN received")
            counter+=1
            if (time.time() - start_time) > 1 :
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()

            # get layer1 data
            layer1 = in_nn.getFirstLayerInt32()
            # reshape to numpy array
            lay1 = np.asarray(layer1, dtype=np.int32).reshape((nn_shape, nn_shape))
            output_colors = decode_deeplabv3p(lay1)

            if frame is not None:
                frame = show_deeplabv3p(output_colors, frame)
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                cv2.imshow("nn", frame)

            # You can add custom code here, for example depth averaging
            if depth_frame is not None:
                depth_overlay = lay1*depth_frame
                depthFrameOverlay = cv2.normalize(depth_overlay, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameOverlay = cv2.equalizeHist(depthFrameOverlay)
                depthFrameOverlay = cv2.applyColorMap(depthFrameOverlay, cv2.COLORMAP_HOT)
                cv2.imshow("depth_overlay", depthFrameOverlay)

        if cv2.waitKey(1) == ord('q'):
            break
