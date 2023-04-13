import depthai as dai
import numpy as np
import utils
import cv2

def create_stereo(pipeline, name, leftIn, rightIn, syncedOutputs=False, rectLeft=False, rectRight=False):
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedLeft = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedRight = pipeline.create(dai.node.XLinkOut)
    xoutDisparity = pipeline.create(dai.node.XLinkOut)
    stereo = pipeline.create(dai.node.StereoDepth)

    xoutLeft.setStreamName(name + "-left")
    xoutRight.setStreamName(name + "-right")
    xoutRectifiedLeft.setStreamName(name  + "-rectified_left")
    xoutRectifiedRight.setStreamName(name + "-rectified_right")
    xoutDisparity.setStreamName(name + "-disparity")

    # Linking
    leftIn.link(stereo.left)
    rightIn.link(stereo.right)
    if syncedOutputs:
        stereo.syncedLeft.link(xoutLeft.input)
        stereo.syncedRight.link(xoutRight.input)
    stereo.disparity.link(xoutDisparity.input)
    if rectLeft:
        stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    if rectRight:
        stereo.rectifiedRight.link(xoutRectifiedRight.input)
    return stereo, [xoutDisparity.getStreamName(), xoutRectifiedLeft.getStreamName(),
                    xoutRectifiedRight.getStreamName(), xoutLeft.getStreamName(),
                    xoutRight.getStreamName()]

def create_mesh_on_host(calibData, leftSocket, rightSocket, resolution, vertical=False):
    width = resolution[0]
    height = resolution[1]

    M1 = np.array(calibData.getCameraIntrinsics(leftSocket, width, height))
    d1 = np.array(calibData.getDistortionCoefficients(leftSocket))
    M2 = np.array(calibData.getCameraIntrinsics(rightSocket, width, height))
    d2 = np.array(calibData.getDistortionCoefficients(rightSocket))

    T = np.array(calibData.getCameraTranslationVector(leftSocket, rightSocket, False))
    R = np.array(calibData.getCameraRotationMatrix(leftSocket, rightSocket))
    R1, R2 = utils.stereoRectify(R, T)

    T2 = np.array(calibData.getCameraTranslationVector(leftSocket, rightSocket, True))

    baseline = abs(T2[1])*10
    focal = M1[1][1]
    focalScaleFactor = baseline * focal * 32

    def calc_fov_D_H_V(f, w, h):
        return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))

    M1_focal = M1
    M1_focal = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M1, d1, resolution, R1)
    print(calc_fov_D_H_V(M1_focal[0][0], width, height))
    focalVer = M1_focal[1][1]

    mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(M1, d1, R1, M1_focal, resolution, cv2.CV_32FC1)
    mapXV, mapYV = cv2.fisheye.initUndistortRectifyMap(M2, d2, R2, M1_focal, resolution, cv2.CV_32FC1)
    if vertical:
        mapXL_rot, mapYL_rot = utils.rotate_mesh_90_ccw(mapXL, mapYL)
        mapXV_rot, mapYV_rot = utils.rotate_mesh_90_ccw(mapXV, mapYV)
    else:
        mapXL_rot, mapYL_rot = mapXL, mapYL
        mapXV_rot, mapYV_rot = mapXV, mapYV
    leftMeshRot, verticalMeshRot = utils.downSampleMesh(mapXL_rot, mapYL_rot, mapXV_rot, mapYV_rot)

    meshLeft = list(leftMeshRot.tobytes())
    meshVertical = list(verticalMeshRot.tobytes())

    return meshLeft, meshVertical, focalScaleFactor