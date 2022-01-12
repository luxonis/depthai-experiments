#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

'''
Deeplabv3 person running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 deeplabv3_person_256.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob taken from the great PINTO zoo

git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob

'''

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input 
nn_path = args.nn_model 

nn_shape = 256
if '513' in nn_path:
    nn_shape = 513

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

pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape,nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(40)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline.getOpenVINOVersion()) as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if cam_source != "rgb" and not depth_enabled:
        raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
    device.startPipeline(pipeline)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_nn_input = q_nn_input.get()
        in_nn = q_nn.get()

        if in_nn_input is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
            frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if in_nn is not None:
            # print("NN received")
            layers = in_nn.getAllLayers()

            if not layer_info_printed:
                for layer_nr, layer in enumerate(layers):
                    print(f"Layer {layer_nr}")
                    print(f"Name: {layer.name}")
                    print(f"Order: {layer.order}")
                    print(f"dataType: {layer.dataType}")
                    dims = layer.dims[::-1] # reverse dimensions
                    print(f"dims: {dims}")
                layer_info_printed = True

            # get layer1 data
            layer1 = in_nn.getLayerInt32(layers[0].name)
            # reshape to numpy array
            dims = layer.dims[::-1]
            lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)

            output_colors = decode_deeplabv3p(lay1)

            if frame is not None:
                frame = show_deeplabv3p(output_colors, frame)
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                cv2.imshow("nn_input", frame)

        counter+=1
        if (time.time() - start_time) > 1 :
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break
