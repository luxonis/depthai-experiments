import depthai as dai
import numpy as np
import sys
from pathlib import Path

# Connect Device
with dai.Device() as device:
    calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
    if len(sys.argv) > 1:
        calibFile = sys.argv[1]

    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)

    M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
    print("RGB Camera Default intrinsics...")
    print(M_rgb)
    print(width)
    print(height)

    if "OAK-1" in calibData.getEepromData().boardName or "BW1093OAK" in calibData.getEepromData().boardName:
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 1280, 720))
        print("RGB Camera resized intrinsics...")
        print(M_rgb)

        D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        print("RGB Distortion Coefficients...")
        [print(name + ": " + value) for (name, value) in
         zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
             [str(data) for data in D_rgb])]

        print(f'RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}')

    else:
        M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        print("RGB Camera Default intrinsics...")
        print(M_rgb)
        print(width)
        print(height)

        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 3840, 2160))
        print("RGB Camera resized intrinsics... 3840 x 2160 ")
        print(M_rgb)


        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 4056, 3040 ))
        print("RGB Camera resized intrinsics... 4056 x 3040 ")
        print(M_rgb)


        M_left, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.LEFT)
        print("LEFT Camera Default intrinsics...")
        print(M_left)
        print(width)
        print(height)

        M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
        print("LEFT Camera resized intrinsics...  1280 x 720")
        print(M_left)


        M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
        print("RIGHT Camera resized intrinsics... 1280 x 720")
        print(M_right)

        D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
        print("LEFT Distortion Coefficients...")
        [print(name+": "+value) for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","τx","τy"],[str(data) for data in D_left])]

        D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
        print("RIGHT Distortion Coefficients...")
        [print(name+": "+value) for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","τx","τy"],[str(data) for data in D_right])]

        print(f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.LEFT)}")

        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())
        M_right = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), 1280, 720))

        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        print("LEFT Camera stereo rectification matrix...")
        print(H_left)

        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
        print("RIGHT Camera stereo rectification matrix...")
        print(H_right)

        lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
        print("Transformation matrix of where left Camera is W.R.T right Camera's optical center")
        print(lr_extrinsics)

        l_rgb_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RGB))
        print("Transformation matrix of where left Camera is W.R.T RGB Camera's optical center")
        print(l_rgb_extrinsics)