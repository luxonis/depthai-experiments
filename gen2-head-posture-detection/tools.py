import numpy as np
import cv2
import math

# World Coordinate System (UVW): fill in the 3D reference point, the model reference http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],    # Upper left corner of left eyebrow
                         [1.330353, 7.122144, 6.903745],    # Left eyebrow right corner
                         [-1.330353, 7.122144, 6.903745],   # Right eyebrow left corner
                         [-6.825897, 6.760612, 4.402142],   # Upper right corner of right eyebrow
                         [5.311432, 5.485328, 3.987654],    # Upper left corner of left eye
                         [1.789930, 5.393625, 4.413414],    # Upper right corner of left eye
                         [-1.789930, 5.393625, 4.413414],   # Upper left corner of right eye
                         [-5.311432, 5.485328, 3.987654],   # Upper right corner of right eye
                         [2.005628, 1.409845, 6.165652],    # Upper left corner of nose
                         [-2.005628, 1.409845, 6.165652],   # Upper right corner of nose
                         [2.774015, -2.080775, 5.048531],   # Upper left corner of mouth
                         [-2.774015, -2.080775, 5.048531],  # Upper right corner of mouth
                         [0.000000, -3.116408, 6.097667],   # Lower corner of mouth
                         [0.000000, -7.415691, 4.070434]])  # Chin angle

# Camera coordinate system (XYZ): add camera internal parameters
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]# Equivalent to matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
# Image center coordinate system (uv): camera distortion parameters [k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# Pixel coordinate system (xy): fill in the eigen and distortion coefficient of the cam
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)



# Reproject the world coordinate axis of the 3D point to verify the resulting pose
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

def get_head_pose(shape):# Head pose estimation
    # (Set of pixel coordinates) Fill in the 2D reference point, and the notes follow https://ibug.doc.ic.ac.uk/resources/300-W/
    # 0 Left eyebrow upper left corner
    # 1 Left eyebrow right corner
    # 2 Right eyebrow upper left corner
    # 3 Right eyebrow upper right corner
    # 4 Left eye upper left corner
    # 5 Left eye upper right corner
    # 6 Right eye upper left corner
    # 7 Upper right corner of the right eye
    # 8 Upper left corner of the nose
    # 9 Upper right corner of the nose
    # 10 Upper left corner
    # 11 Upper right corner of the mouth
    # 12 Lower central corner of the mouth
    # 13 Chin corner
    image_pts = np.float32([shape[0], shape[1], shape[2], shape[3], shape[4],
                            shape[5], shape[6], shape[7], shape[8], shape[9],
                            shape[10], shape[11], shape[12], shape[13]])
    # solvePnP Calculate the pose-solve the rotation and translation matrix：
    # rotation_vec Represents the rotation matrix，translation_vec Represents the translation matrix，cam_matrix Corresponds to K matrix，dist_coeffs Corresponds to D matrix。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPointsRe-projection error: the distance between the original 2d point and the reprojected 2d point (input 3d point, camera internal parameters, camera distortion, r, t, output reprojected 2d point)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# Display in 8 rows and 2 columns

    # Calculate Euler angle calc euler angle
    # reference https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)# Rodriguez formula (convert the rotation matrix to a rotation vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))# Horizontal splicing, vconcat vertical splicing
    # decomposeProjectionMatrix Decompose projection matrix into rotation matrix and camera matrix
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
 
 
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle, pitch, yaw, roll# Projection error, Euler angle