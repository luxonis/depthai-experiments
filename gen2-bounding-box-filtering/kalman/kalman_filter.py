import depthai as dai
import blobconverter
import cv2
import time
import numpy as np
import pandas as pd
import imageio


label_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# For OAK-D camera
df = pd.read_csv('errors.csv')
error_poly = np.poly1d(np.polyfit(df['Ground truth'], df['OAK-D Error [%]'], 1))


class KalmanFilter(object):
    def __init__(self, acc_std, meas_std, z, time):
        self.dim_z = len(z)
        self.time = time
        self.acc_std = acc_std
        self.meas_std = meas_std

        # the observation matrix 
        self.H = np.eye(self.dim_z, 3 * self.dim_z) 

        self.x = np.vstack((z, np.zeros((2 * self.dim_z, 1))))
        self.P = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        i, j = np.indices((3 * self.dim_z, 3 * self.dim_z))
        self.P[(i - j) % self.dim_z == 0] = 1e5 # initial vector is a guess -> high estimate uncertainty


    def predict(self, dt):
        # the state transition matrix -> assuming acceleration is constant
        F = np.eye(3 * self.dim_z)
        np.fill_diagonal(F[:2*self.dim_z, self.dim_z:], dt)
        np.fill_diagonal(F[:self.dim_z, 2*self.dim_z:], dt ** 2 / 2)

        # the process noise matrix
        A = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        np.fill_diagonal(A[2*self.dim_z:, 2*self.dim_z:], 1)
        Q = self.acc_std ** 2 * F @ A @ F.T 

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    
    def update(self, z):
        if z is None: return

        # the measurement uncertainty
        R = self.meas_std ** 2 * np.eye(self.dim_z)

        # the Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(3 * self.dim_z)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T


# Create pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
detection_network = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
object_tracker = pipeline.create(dai.node.ObjectTracker)

xout_rgb = pipeline.create(dai.node.XLinkOut)
tracker_out = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName('rgb')
tracker_out.setStreamName('tracklets')

cam_rgb.setPreviewSize(300, 300)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())

detection_network.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=5))
detection_network.setConfidenceThreshold(0.7)
detection_network.input.setBlocking(False)
detection_network.setBoundingBoxScaleFactor(0.5)
detection_network.setDepthLowerThreshold(100)
detection_network.setDepthUpperThreshold(5000)

object_tracker.setDetectionLabelsToTrack([15])  # track only person
object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam_rgb.preview.link(detection_network.input)
object_tracker.passthroughTrackerFrame.link(xout_rgb.input)
object_tracker.out.link(tracker_out.input)

detection_network.passthrough.link(object_tracker.inputTrackerFrame)
detection_network.passthrough.link(object_tracker.inputDetectionFrame)
detection_network.out.link(object_tracker.inputDetections)
stereo.depth.link(detection_network.inputDepth)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    q_rgb  = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
    q_tracklets = device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)

    kalman_filters = {}
    # frames = []

    while(True):
        frame = q_rgb.get().getCvFrame()
        tracks = q_tracklets.get().tracklets

        for t in tracks:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            x_space = t.spatialCoordinates.x
            y_space = t.spatialCoordinates.y
            z_space = t.spatialCoordinates.z

            meas_vec_bbox = np.array([[x_center], [y_center], [width], [height]])
            meas_vec_space = np.array([[x_space], [y_space], [z_space]])

            current_time = time.monotonic()
            meas_std_space =  error_poly(z_space) * z_space / 100

            if t.status.name == 'NEW':
                dt = 1e-2

                # Adjust these parameters
                acc_std_space = 10
                acc_std_bbox = 0.1
                meas_std_bbox = 0.05

                kalman_filters[t.id] = {'bbox': KalmanFilter(meas_std_bbox, acc_std_bbox, meas_vec_bbox, current_time),
                                        'space': KalmanFilter(meas_std_space, acc_std_space, meas_vec_space, current_time)}

            else:
                dt = current_time - kalman_filters[t.id]['bbox'].time
                kalman_filters[t.id]['space'].meas_std = meas_std_space

                if t.status.name != 'TRACKED':
                    meas_vec_bbox = None
                    meas_vec_space = None

                if z_space == 0:
                    meas_vec_space = None


            kalman_filters[t.id]['bbox'].predict(dt)
            kalman_filters[t.id]['bbox'].update(meas_vec_bbox)

            kalman_filters[t.id]['space'].predict(dt)
            kalman_filters[t.id]['space'].update(meas_vec_space)

            kalman_filters[t.id]['bbox'].time = current_time
            kalman_filters[t.id]['space'].time = current_time

            vec_bbox = kalman_filters[t.id]['bbox'].x
            vec_space = kalman_filters[t.id]['space'].x

            x1_filter = int(vec_bbox[0] - vec_bbox[2] / 2)
            x2_filter = int(vec_bbox[0] + vec_bbox[2] / 2)
            y1_filter = int(vec_bbox[1] - vec_bbox[3] / 2)
            y2_filter = int(vec_bbox[1] + vec_bbox[3] / 2)

            try:
                label = label_map[t.label]
            except:
                label = t.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f'ID: {[t.id]}', (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))
            cv2.rectangle(frame, (x1_filter, y1_filter), (x2_filter, y2_filter), (0, 0, 255), 2)

            cv2.putText(frame, f'X: {int(x_space)} mm', (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f'Y: {int(y_space)} mm', (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f'Z: {int(z_space)} mm', (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.putText(frame, f'X: {int(vec_space[0])} mm', (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f'Y: {int(vec_space[1])} mm', (x1 + 10, y1 + 125), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f'Z: {int(vec_space[2])} mm', (x1 + 10, y1 + 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

        cv2.imshow('tracker', frame)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frames.append(rgb_frame)

        if cv2.waitKey(1) == ord('q'):
            break

# imageio.mimsave('demo.gif', frames, fps=30)
