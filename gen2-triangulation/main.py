import blobconverter
import numpy as np
import math
import cv2
import depthai as dai
import re


VISUALIZE = False

if VISUALIZE:
    import open3d as o3d

openvinoVersion = "2021.3"
p = dai.Pipeline()
p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)

# Set resolution of mono cameras
resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# THE_720_P => 720
resolution_num = int(re.findall("\d+", str(resolution))[0])

def populate_pipeline(p, name, resolution):
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
    cam.setBoardSocket(socket)
    cam.setResolution(resolution)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(face_manip.inputImage)

    # NN that detects faces in the image
    face_nn = p.create(dai.node.NeuralNetwork)
    face_nn.setBlobPath(str(blobconverter.from_zoo("face-detection-retail-0004", shaves=6, version=openvinoVersion)))
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
    image_manip_script.setScript("""
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
    landmarks_nn.setBlobPath(str(blobconverter.from_zoo("landmarks-regression-retail-0009", shaves=6, version=openvinoVersion)))
    manip_crop.out.link(landmarks_nn.input)

    landmarks_nn_xout = p.createXLinkOut()
    landmarks_nn_xout.setStreamName("landmarks_" + name)
    landmarks_nn.out.link(landmarks_nn_xout.input)

populate_pipeline(p, "right", resolution)
populate_pipeline(p, "left", resolution)

class StereoInference:
    def __init__(self, device, resolution_num, mono_width, mono_heigth) -> None:
        calibData = device.readCalibration()
        lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT, useSpecTranslation=True))
        translation_vector = lr_extrinsics[0:3,3:4].flatten()
        baseline_cm = math.sqrt(np.sum(translation_vector ** 2))
        self.baseline = baseline_cm * 10 # to get in mm

        self.hfov = calibData.getFov(dai.CameraBoardSocket.LEFT)
        # Cropped frame shape
        self.mono_width = mono_width
        self.mono_heigth = mono_heigth
        # Original mono frames shape
        self.original_heigth = resolution_num
        self.original_width = 640 if resolution_num == 400 else 1280

        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 to get the correct disparity.
        self.resize_factor = self.original_heigth / self.mono_heigth

        self.focal_length_pixels = self.get_focal_length_pixels(self.original_width, self.hfov)
        print('focal', self.focal_length_pixels)

    def get_focal_length_pixels(self, pixel_width, hfov):
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi/180)

    def calculate_depth(self, disparity_pixels):
        if disparity_pixels == 0: return 9999999999999 # To avoid dividing by 0 error
        return self.focal_length_pixels * self.baseline / disparity_pixels

    def calculate_distance(self, c1, c2):
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from 720x720 (by ImageManip),
        # so we need to multiple coords by 2.4 (if using 720P resolution) to get the correct disparity.
        c1 = np.array(c1) * self.resize_factor
        c2 = np.array(c2) * self.resize_factor

        x_delta = c1[0] - c2[0]
        y_delta = c1[1] - c2[1]
        return math.sqrt(x_delta ** 2 + y_delta ** 2)

    def calc_angle(self, offset):
            return math.atan(math.tan(self.hfov / 2.0) * offset / (self.mono_width / 2.0))

    def calc_spatials(self, coords, depth):
        x, y = coords
        bb_x_pos = x - self.mono_width / 2
        bb_y_pos = y - self.mono_heigth / 2

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = depth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)
        # print(f"x {x}, y {y}, z {z}")
        return [x,y,z]

# Pipeline is defined, now we can connect to the device
with dai.Device(p.getOpenVINOVersion()) as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(p)
    # Set device log level - to see logs from the Script node
    device.setLogLevel(dai.LogLevel.WARN)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    stereoInference = StereoInference(device, resolution_num, mono_width=300, mono_heigth=300)

    if VISUALIZE:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    # Start pipeline
    queues = []
    for name in ["left", "right"]:
        queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
        queues.append(device.getOutputQueue(name="config_"+name, maxSize=4, blocking=False))
    while True:
        lr_landmarks = {}
        disparity_frame = None
        for i in range(2):
            name = "left" if i == 1 else "right"
            # 300x300 Mono image frame
            frame = queues[i*4].get().getCvFrame()

            # Combine the two mono frames
            if i == 0: disparity_frame = frame
            else: disparity_frame = cv2.addWeighted(disparity_frame, 0.5, frame,0.5, 0)

            # Cropped+streched (48x48) mono image frame
            cropped_frame = queues[i*4 + 1].get().getCvFrame()

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
                landmarks_layer = queues[i*4 + 2].get().getFirstLayerFp16()
                landmarks = np.array(landmarks_layer).reshape(5, 2)

                landmarks_xy = []
                for landmark in landmarks:
                    cv2.circle(cropped_frame, (int(48*landmark[0]), int(48*landmark[1])), 3, (0, 255, 0))
                    x = int((landmark[0] * width + inConfig.getCropXMin()) * 300)
                    y = int((landmark[1] * height + inConfig.getCropYMin()) * 300)
                    landmarks_xy.append((x,y))
                    cv2.circle(frame, (x, y), 3, (0,255,0))

                lr_landmarks[name] = landmarks_xy
            # Display both mono/cropped frames
            cv2.imshow("mono_"+name, frame)
            cv2.imshow("crop_"+name, cropped_frame)

        # 3D visualization
        if len(lr_landmarks) == 2:
            spatials = []
            for i in range(5):
                coords1 = lr_landmarks['right'][i]
                coords2 = lr_landmarks['left'][i]

                # Visualize disparity line frame
                cv2.circle(disparity_frame, coords1, 3, (0,255,0))
                cv2.circle(disparity_frame, coords2, 3, (255,0,0))
                cv2.line(disparity_frame, coords1, coords2, (0,0,255), 1)
                cv2.imshow("Disparity frame", disparity_frame)

                disparity = stereoInference.calculate_distance(coords1, coords2)
                depth = stereoInference.calculate_depth(disparity)
                # print(f"Disp {disparity}, depth {depth}")
                spatials.append(stereoInference.calc_spatials(coords1, depth))

            if VISUALIZE:
                # For 3d point projection.
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(spatials)
                vis.clear_geometries()
                vis.add_geometry(pcl)
            else:
                for i, s in enumerate(spatials):
                    print(f"[Landmark {i}] X: {s[0]}, Y: {s[1]}, Z: {s[2]}")
            lr_landmarks = {}

        if VISUALIZE:
            vis.poll_events()
            vis.update_renderer()

        if cv2.waitKey(1) == ord('q'):
            break
