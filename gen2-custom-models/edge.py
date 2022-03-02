import numpy as np
import cv2
import depthai as dai

SHAPE = 300

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = p.create(dai.node.ColorCamera)
camRgb.setPreviewSize(SHAPE, SHAPE)
camRgb.setInterleaved(False)

# NN that detects faces in the image
nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath("models/edge_simplified_openvino_2021.4_6shave.blob")
camRgb.preview.link(nn.input)

# Send bouding box from the NN to the host via XLink
nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

rgb_xout = p.create(dai.node.XLinkOut)
rgb_xout.setStreamName("rgb")
camRgb.preview.link(rgb_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    shape = (3, SHAPE, SHAPE)

    def get_frame(imfFrame, shape):
        return np.array(imfFrame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)

    while True:
        cv2.imshow("Laplacian edge detection", get_frame(qNn.get(), shape))
        cv2.imshow("Color", qCam.get().getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
