
import cv2
import numpy as np
from openvino.inference_engine import IECore

def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    return frame[0:height, delta:width-delta]


model_xml = 'facial_cartoonization_256x256.xml'
model_bin = "facial_cartoonization_256x256.bin"
shape = (256, 256)

ie = IECore()
print("Available devices:", ie.available_devices)
net = ie.read_network(model=model_xml, weights=model_bin)
input_blob = next(iter(net.input_info))
# You can select device_name="CPU" to run on CPU
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# Get video from the computers webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, raw_image = cam.read()
    if not ret:
        continue
    image = crop_to_square(raw_image)
    image = cv2.resize(image, shape)
    cv2.imshow('USB Camera', image)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = image.transpose((0, 3, 1, 2))
    image = image / 127.5 - 1.0

    # Do the inference on the MYRIAD device
    output = exec_net.infer(inputs={input_blob: image})
    output = (output['up4'] + 1) * 127.5
    output = output.transpose((0, 2, 3, 1))[0]
    output = output.astype(np.uint8)
    cv2.imshow('Output', output)

    if cv2.waitKey(1) == ord('q'):
        break
