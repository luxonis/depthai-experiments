import numpy as np
import cv2
import depthai as dai

SHAPE = 300

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = p.createColorCamera()
camRgb.setPreviewSize(SHAPE, SHAPE)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

left = p.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
manipLeft = p.create(dai.node.ImageManip)
manipLeft.initialConfig.setResize(300, 300)
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manipLeft.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
left.out.link(manipLeft.inputImage)

right = p.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
manipRight = p.create(dai.node.ImageManip)
manipRight.initialConfig.setResize(300, 300)
# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
manipRight.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
right.out.link(manipRight.inputImage)

script = p.create(dai.node.Script)
camRgb.preview.link(script.inputs['color'])
manipLeft.out.link(script.inputs['left'])
manipRight.out.link(script.inputs['right'])
script.setScript("""
while True:
    left = node.io['left'].get().getData()
    right = node.io['right'].get().getData()
    color = node.io['color'].get().getData()
    nndata = NNData(810000) # Three images of with 3x300x300 size
    nndata.setLayer("img1", left)
    nndata.setLayer("img2", color)
    nndata.setLayer("img3", right)
    node.io['out'].send(nndata)
""")

# NN that detects faces in the image
nn = p.createNeuralNetwork()
nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
nn.setNumInferenceThreads(2)
script.outputs['out'].link(nn.input)

# Send bouding box from the NN to the host via XLink
nn_xout = p.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    shape = (3, SHAPE, SHAPE * 3)

    while True:
        inNn = np.array(qNn.get().getData())
        frame = inNn.view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()

        cv2.imshow("Concat", frame)

        if cv2.waitKey(1) == ord('q'):
            break
