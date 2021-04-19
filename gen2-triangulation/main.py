# import math
from pathlib import Path
import numpy as np
import math
from visualizer import initialize_OpenGL, get_vector_direction, get_vector_intersection, start_OpenGL
import cv2
import depthai as dai

p = dai.Pipeline()

left_camera_position = (0.107, -0.038, 0.008)
right_camera_position = (0.109, 0.039, 0.008)
cameras = ((0.107, -0.038, 0.008), (0.109, 0.039, 0.008))

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
    face_nn.setBlobPath(str(Path("models/face-detection-retail-0004_2021.3_6shaves.blob").resolve().absolute()))
    face_manip.out.link(face_nn.input)

    # Send bouding box from the NN to host via XLink
    nn_xout = p.createXLinkOut()
    nn_xout.setStreamName("nn_" + name)
    face_nn.out.link(nn_xout.input)

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
        node.io['to_manip'].send(cfg)
        #node.warn(f"1 from nn_in: {x_min}, {y_min}, {x_max}, {y_max}")
""")

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    manip_crop = p.create(dai.node.ImageManip)
    face_nn.passthrough.link(manip_crop.inputImage)
    image_manip_script.outputs['to_manip'].link(manip_crop.inputConfig)
    manip_crop.setWaitForConfigInput(False)

    crop_xout = p.createXLinkOut()
    crop_xout.setStreamName("crop_" + name)
    manip_crop.out.link(crop_xout.input)

    # Second NN that detcts landmarks from the cropped 48x48 face
    landmarks_nn = p.createNeuralNetwork()
    landmarks_nn.setBlobPath(str(Path("models/landmarks-regression-retail-0009_2021.3_6shaves.blob").resolve().absolute()))
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
with dai.Device(p) as device:
    # Set device log level - to see logs from the Script node
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    # Start pipeline
    device.startPipeline()
    queues = []
    for name in ["left", "right"]:
        queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="nn_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
    while True:
        lr_landmarks = []
        for i in range(2):
            name = "left" if i == 1 else "right"
            # 300x300 Mono image frame
            inMono = queues[i*4].get()
            frame = inMono.getCvFrame()

            # Cropped+streched (48x48) mono image frame
            inCrop = queues[i*4 + 1] .get()
            cropped_frame = inCrop.getCvFrame()

            # Face detection data from the first NN
            inNn = queues[i*4 + 2].get()
            nn_arr=np.array(inNn.getFirstLayerFp16())
            nn_data = nn_arr[2:7]
            face_xmin, face_ymin, face_xmax, face_ymax = nn_data[1:5]
            cv2.rectangle(frame, (int(300 * face_xmin), int(300 * face_ymin)), (int(300 * face_xmax), int(300 * face_ymax)), (0, 255, 0), 2)

            # Facial landmarks from the second NN
            inLandmarks = queues[i*4 + 3].get()
            landmarks_layer = inLandmarks.getFirstLayerFp16()
            landmarks = np.array(landmarks_layer).reshape(5, 2)
            nn_w = nn_data[3]-nn_data[1]
            nn_h = nn_data[4]-nn_data[2]

            lr_landmarks.append(list(map(get_landmark_3d, landmarks)))
            for landmark in landmarks:
                get_landmark_3d(landmark)
                w = landmark[0] * nn_w + nn_data[1]
                h = landmark[1] * nn_h + nn_data[2]
                cv2.circle(cropped_frame, (int(48*landmark[0]), int(48*landmark[1])), 3, (0, 255, 0))
                cv2.circle(frame, (int(300 * w), int(300 * h)), 3, (0,255,0))

            # Display both mono/cropped frames
            cv2.imshow("mono_"+name, frame)
            cv2.imshow("crop_"+name, cropped_frame)

        # 3D visualization
        if len(lr_landmarks[0]) > 0 and len(lr_landmarks[1]) > 0:
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
