import numpy as np
import math

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def angle(v1, v2, ax):
    u1 = v1 - v1 @ ax * ax
    u2 = v2 - v2 @ ax * ax

    a = np.arccos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))
    dir = -1 if np.cross(u1, u2) @ ax < 0 else 1
    return a * dir

def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = (R_z @ (R_y @ R_x))

    return R


def stereoRectify(R, T):

    om = rotationMatrixToEulerAngles(R)
    om = om * -0.5
    r_r = eulerAnglesToRotationMatrix(om)
    t = r_r @ T

    idx = 0 if abs(t[0]) > abs(t[1]) else 1

    c = t[idx]
    nt = np.linalg.norm(t)
    uu = np.zeros(3)
    uu[idx] = 1 if c > 0 else -1

    ww = np.cross(t, uu)
    nw = np.linalg.norm(ww)

    if nw > 0:
        scale = math.acos(abs(c)/nt)/nw
        ww = ww * scale

    wR = eulerAnglesToRotationMatrix(ww)
    R1 = wR @ np.transpose(r_r)
    R2 = wR @ r_r

    return R1, R2


meshCellSize = 16


def downSampleMesh(mapXL, mapYL, mapXR, mapYR):
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def rotate_mesh_90_ccw(map_x, map_y):
    direction = 1
    map_x_rot = np.rot90(map_x, direction)
    map_y_rot = np.rot90(map_y, direction)
    return map_x_rot, map_y_rot

def rotate_mesh_90_cw(map_x, map_y):
    direction = -1
    map_x_rot = np.rot90(map_x, direction)
    map_y_rot = np.rot90(map_y, direction)
    return map_x_rot, map_y_rot