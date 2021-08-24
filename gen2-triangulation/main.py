# import math
from pathlib import Path

import blobconverter
import numpy as np
import math
from visualizer import initialize_OpenGL, get_vector_direction, get_vector_intersection, start_OpenGL
import cv2
import depthai as dai

p = dai.Pipeline()

left_camera_position = (0.107, -0.038, 0.008)
right_camera_position = (0.109, 0.039, 0.008)
cameras = (left_camera_position, right_camera_position)

def populatePipeline(p, name):
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
    cam.setBoardSocket(socket)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(face_manip.inputImage)

    # NN that detects faces in the image
    face_nn = p.create(dai.node.NeuralNetwork)
    face_nn.setBlobPath(str(blobconverter.from_zoo("face-detection-retail-0004", shaves=6)))
    face_manip.out.link(face_nn.input)

    # Send mono frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("mono_" + name)
    face_nn.passthrough.link(cam_xout.input)

    # Script node will take the output from the NN as an input, get the first bounding box
    # and if the confidence is greater than 0.2, script will send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    image_manip_script.inputs['nn_in'].setBlocking(False)
    image_manip_script.inputs['nn_in'].setQueueSize(1)
    face_nn.out.link(image_manip_script.inputs['nn_in'])
    image_manip_script.setScriptData("""
while True:
    nn_in = node.io['nn_in'].get()
    nn_data = nn_in.getFirstLayerFp16()

    conf=nn_data[2]
    if 0.2<conf:
        x_min=nn_data[3]
        y_min=nn_data[4]
        x_max=nn_data[5]
        y_max=nn_data[6]
        cfg = ImageManipConfig()
        cfg.setCropRect(x_min, y_min, x_max, y_max)
        cfg.setResize(48, 48)
        cfg.setKeepAspectRatio(False)
        node.io['to_manip'].send(cfg)
        #node.warn(f"1 from nn_in: {x_min}, {y_min}, {x_max}, {y_max}")
""")

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    manip_crop = p.create(dai.node.ImageManip)
    face_nn.passthrough.link(manip_crop.inputImage)
    image_manip_script.outputs['to_manip'].link(manip_crop.inputConfig)
    manip_crop.initialConfig.setResize(48, 48)
    manip_crop.setWaitForConfigInput(False)

    # Send ImageManipConfig to host so it can visualize the landmarks
    config_xout = p.create(dai.node.XLinkOut)
    config_xout.setStreamName("config_" + name)
    image_manip_script.outputs['to_manip'].link(config_xout.input)

    crop_xout = p.createXLinkOut()
    crop_xout.setStreamName("crop_" + name)
    manip_crop.out.link(crop_xout.input)

    # Second NN that detcts landmarks from the cropped 48x48 face
    landmarks_nn = p.createNeuralNetwork()
    landmarks_nn.setBlobPath(str(blobconverter.from_zoo("landmarks-regression-retail-0009", shaves=6)))
    manip_crop.out.link(landmarks_nn.input)

    landmarks_nn_xout = p.createXLinkOut()
    landmarks_nn_xout.setStreamName("landmarks_" + name)
    landmarks_nn.out.link(landmarks_nn_xout.input)


populatePipeline(p, "right")
populatePipeline(p, "left")

def get_landmark_3d(landmark):
    focal_length = 842
    landmark_norm = 0.5 - np.array(landmark)

    # image size
    landmark_image_coord = landmark_norm * 640

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D

initialize_OpenGL()

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(p)
    # Set device log level - to see logs from the Script node
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    # Start pipeline
    device.startPipeline()
    queues = []
    for name in ["left", "right"]:
        queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="config_"+name, maxSize=4, blocking=False))
    while True:
        lr_landmarks = []
        for i in range(2):
            name = "left" if i == 1 else "right"
            # 300x300 Mono image frame
            inMono = queues[i*4].get()
            frame = inMono.getCvFrame()

            # Cropped+streched (48x48) mono image frame
            inCrop = queues[i*4 + 1].get()
            cropped_frame = inCrop.getCvFrame()

            inConfig = queues[i*4 + 3].tryGet()
            if inConfig is not None:
                xmin = int(300 * inConfig.getCropXMin())
                ymin = int(300 * inConfig.getCropYMin())
                xmax = int(300 * inConfig.getCropXMax())
                ymax = int(300 * inConfig.getCropYMax())
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                width = inConfig.getCropXMax()-inConfig.getCropXMin()
                height = inConfig.getCropYMax()-inConfig.getCropYMin()

                # Facial landmarks from the second NN
                inLandmarks = queues[i*4 + 2].get()
                landmarks_layer = inLandmarks.getFirstLayerFp16()
                landmarks = np.array(landmarks_layer).reshape(5, 2)

                lr_landmarks.append(list(map(get_landmark_3d, landmarks)))
                for landmark in landmarks:
                    cv2.circle(cropped_frame, (int(48*landmark[0]), int(48*landmark[1])), 3, (0, 255, 0))
                    w = landmark[0] * width + inConfig.getCropXMin()
                    h = landmark[1] * height + inConfig.getCropYMin()
                    cv2.circle(frame, (int(w * 300), int(h * 300)), 3, (0,255,0))

            # Display both mono/cropped frames
            cv2.imshow("mono_"+name, frame)
            cv2.imshow("crop_"+name, cropped_frame)

        # 3D visualization
        if len(lr_landmarks) == 2 and len(lr_landmarks[0]) > 0 and len(lr_landmarks[1]) > 0:
                mid_intersects = []
                for i in range(5):
                    left_vector = get_vector_direction(left_camera_position, lr_landmarks[0][i])
                    right_vector = get_vector_direction(right_camera_position, lr_landmarks[1][i])
                    intersection_landmark = get_vector_intersection(left_vector, left_camera_position, right_vector,
                                                                    right_camera_position)
                    mid_intersects.append(intersection_landmark)

                start_OpenGL(mid_intersects, cameras, lr_landmarks[0], lr_landmarks[1])

        if cv2.waitKey(1) == ord('q'):
            break
