import cv2
import depthai as dai
import numpy as np
import argparse
import time

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/selfie_segmentation_landscape_openvino_2021.4_6shave_RGB_interleaved.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

NN_W, NN_H = 256, 144


# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(NN_W,NN_H)
cam.setInterleaved(True)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.preview.link(detection_nn.input)
cam.setFps(50)

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

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False

while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    frame = in_nn_input.getCvFrame()
    lay1 = in_nn.getFirstLayerFp16()
    pred = np.array(lay1, dtype=np.float16).reshape((NN_H, NN_W))
    #pred = np.transpose(pred, (1,0))

    condition = np.stack([pred > 0.15] * 3, axis = 2)
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = (255, 255, 255)
    output_image = np.where(condition, frame, bg_image)
    output_image = output_image.astype(np.uint8)

    # transpose
    #output_image = np.transpose(output_image, (2, 0, 1))
    #frame = np.transpose(frame, (2, 0, 1))


    color_black, color_white = (0, 0, 0), (255, 255, 255)
    label_fps = "Fps: {:.2f}".format(fps)
    (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
    cv2.rectangle(output_image, (0, output_image.shape[0] - h1 - 6), (w1 + 2, output_image.shape[0]), color_white, -1)
    cv2.putText(output_image, label_fps, (2, output_image.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                0.4, color_black)


    cv2.imshow("nn_input", frame)
    cv2.imshow("result", output_image)

    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()


    if cv2.waitKey(1) == ord('q'):
        break