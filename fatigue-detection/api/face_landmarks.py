import math

import cv2
import numpy as np
from scipy.spatial.distance import euclidean


class FaceLandmarks:
    def __init__(self):
        self.COUNTER = 0
        self.mCOUNTER = 0
        self.hCOUNTER = 0
        self.TOTAL = 0
        self.mTOTAL = 0
        self.hTOTAL = 0

    def run_land68(self, face_frame, nndata, face_coords_float):
        # try:
        out = np.array(nndata.getFirstLayerFp16())
        face_coords = self.normalize_face_coords(face_frame, face_coords_float)
        self.frame = face_frame
        result = self.frame_norm(face_frame, *out)
        eye_left = []
        eye_right = []
        mouth = []
        hand_points = []
        for i in range(72, 84, 2):
            eye_left.append((out[i], out[i + 1]))
        for i in range(84, 96, 2):
            eye_right.append((out[i], out[i + 1]))
        for i in range(96, len(result), 2):
            if i == 100 or i == 116 or i == 104 or i == 112 or i == 96 or i == 108:
                mouth.append(np.array([out[i], out[i + 1]]))

        for i in range(16, 110, 2):
            if i == 16 or i == 60 or i == 72 or i == 90 or i == 96 or i == 108:
                hand_points.append((result[i] + face_coords[0], result[i + 1] + face_coords[1]))

        # Whether dead is tilted downwards
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = self.get_pose_estimation(
            self.frame.shape, np.array(hand_points, dtype='double'))
        ret, pitch, yaw, roll = self.get_euler_angle(rotation_vector)
        if pitch < 0:
            self.hCOUNTER += 1
            if self.hCOUNTER >= 20:
                cv2.putText(self.frame, "SLEEP!!!", (face_coords[0], face_coords[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 3)
        else:
            if self.hCOUNTER >= 3:
                self.hTOTAL += 1
            self.hCOUNTER = 0

        # Whether eyes are closed (based on eye aspect ratio)
        left_ear = self.eye_aspect_ratio(eye_left)
        right_ear = self.eye_aspect_ratio(eye_right)
        ear = (left_ear + right_ear) / 2.0
        if ear < 0.15:  # Eye aspect ratio：0.15
            self.COUNTER += 1
            if self.COUNTER >= 20:
                cv2.putText(self.frame, "SLEEP!!!", (face_coords[0], face_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)
        else:
            # If it is less than the threshold 3 times in a row, it means that an eye blink has been performed
            if self.COUNTER >= 3:  # Threshold: 3
                self.TOTAL += 1
            # Reset the eye frame counter
            self.COUNTER = 0

        # Yawning detection - based on mouth aspect ratio
        mouth_ratio = self.mouth_aspect_ratio(mouth)
        if mouth_ratio > 0.5:
            self.mCOUNTER += 1
        else:
            if self.mCOUNTER >= 3:
                self.mTOTAL += 1
            self.mCOUNTER = 0

        cv2.putText(self.frame, "eye:{:d},mouth:{:d},head:{:d}".format(self.TOTAL, self.mTOTAL, self.hTOTAL), (10, 40),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 0, 0,))
        if self.TOTAL >= 50 or self.mTOTAL >= 15 or self.hTOTAL >= 10:
            cv2.putText(self.frame, "Danger!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def frame_norm(self, frame, *xy_vals):
        height, width = frame.shape[:2]
        result = []
        for i, val in enumerate(xy_vals):
            if i % 2 == 0:
                result.append(max(0, min(width, int(val * width))))
            else:
                result.append(max(0, min(height, int(val * height))))
        return result

    def normalize_face_coords(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[1] - mouth[5])  # 51, 59
        B = np.linalg.norm(mouth[2] - mouth[4])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[3])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def eye_aspect_ratio(self, eye):
        A = euclidean(eye[1], eye[5])
        B = euclidean(eye[2], eye[4])
        C = euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

        # Get rotation vector and translation vector

    def get_pose_estimation(self, img_size, image_points):
        # 3D model points.
        model_points = np.array([
            (0.0, -330.0, -65.0),  # Chin
            (0.0, 0.0, 0.0),  # Nose tip
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera internals
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # print("Camera Matrix :{}".format(camera_matrix))
        # print(image_points)
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # print("Rotation Vector:\n {}".format(rotation_vector))
        # print("Translation Vector:\n {}".format(translation_vector))
        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

    # Convert from rotation vector to Euler angle
    def get_euler_angle(self, rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        # print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        # Unit conversion: convert radians to degrees
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)

        return 0, Y, X, Z
