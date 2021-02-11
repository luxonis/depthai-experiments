#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse

'''
Deeplabv3 person running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 deeplabv3_person_256.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'
'''

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
args = parser.parse_args()

cam_source = args.cam_input 

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    output = output_tensor.reshape(256, 256)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.2,0)



# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setPipelineOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2020_1)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str((Path(__file__).parent / Path('models/deeplabv3p_person.blob.sh13cmx13NCE1')).resolve().absolute()))
detection_nn.setNumPoolFrames(1)
detection_nn.input.setBlocking(False)
# detection_nn.setNumInferenceThreads(1)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(256, 256)
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
    manip.setResize(256, 256)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

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
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)


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
        output_layers = in_nn.getAllLayerNames()
        # print(output_layers)

        layer1 = in_nn.getLayerInt32(output_layers[0])

        lay1 = np.asarray(layer1, dtype=np.int32).reshape((1,256, 256))
        # print(lay1.shape)

        output_colors = decode_deeplabv3p(lay1)

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.imshow("nn_input", frame)

    if cv2.waitKey(1) == ord('q'):
        break
