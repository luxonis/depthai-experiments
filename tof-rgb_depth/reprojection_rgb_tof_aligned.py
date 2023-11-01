import numpy as np
import cv2
import depthai as dai
from numba import jit, prange
print(dai.__version__)

@jit(nopython=True, parallel=True)
def reprojection(depth_image, depth_camera_intrinsics, camera_extrinsics, color_camera_intrinsics, mode, hardware_rectify=True):
    height = len(depth_image)
    width = len(depth_image[0])
    
    image = np.zeros((height, width), np.uint8)
    prev_v_idxs = np.zeros(width, np.uint16)
    for i in prange(0, height):
        prev_u = 0
        for j in prange(0, width):
            d = depth_image[i][j]

            # project 3d point to pixel
            if d == 0:
                prev_v_idxs[j] = 0
                continue

            # convert pixel to 3d point
            x = (j - depth_camera_intrinsics[0][2]) * d / depth_camera_intrinsics[0][0]
            y = (i - depth_camera_intrinsics[1][2]) * d / depth_camera_intrinsics[1][1]
            z = d

            # apply transformation
            if hardware_rectify:
                x1 = x 
                y1 = y 
                z1 = z 
            else:
                x1 = camera_extrinsics[0][0] * x + camera_extrinsics[0][1] * y + camera_extrinsics[0][2] * z + camera_extrinsics[0][3]
                y1 = camera_extrinsics[1][0] * x + camera_extrinsics[1][1] * y + camera_extrinsics[1][2] * z + camera_extrinsics[1][3]
                z1 = camera_extrinsics[2][0] * x + camera_extrinsics[2][1] * y + camera_extrinsics[2][2] * z + camera_extrinsics[2][3]
            
            u = color_camera_intrinsics[0][0] * (x1  / z1) + color_camera_intrinsics[0][2]
            v = color_camera_intrinsics[1][1] * (y1  / z1) + color_camera_intrinsics[1][2]
            int_u = round(u)
            int_v = round(v)
            if int_u >= 0 and int_u < len(image[0]) and int_v >= 0 and int_v < len(image):
                # print(f'j -> {j} => u -> {int_u} and i -> {i} => v -> {int_v}')
                if mode == 2:
                    count = 0
                    while int_v - prev_v_idxs[j] >= 1 and int_v - prev_v_idxs[j] < 3:
                        prev_v_idxs[j] += 1
                        prev_u_loc = prev_u
                        while int_u - prev_u_loc >= 1 and int_u - prev_u_loc < 3:
                            prev_u_loc += 1
                            count += 1
                            if image[prev_v_idxs[j]][prev_u_loc] == 0: 
                                image[prev_v_idxs[j]][prev_u_loc] = z1
                                # if int_u - prev_u_loc != 0 and int_v - prev_v_idxs[j] != 0:
                                    # frameRgb[prev_v_idxs[j]][prev_u_loc] = [255, 255, 255]


                if mode == 3:
                    if image[int_v - 1][int_u] == 0 and image[int_v - 2][int_u] != 0:
                        image[int_v - 1][int_u] = z1
                        # frameRgb[int_v - 1][int_u] = [255, 255, 255]

                    if image[int_v][int_u - 1] == 0 and image[int_v][int_u - 2] != 0:
                        image[int_v][int_u - 1] = z1
                        # frameRgb[int_v][int_u - 1] = [255, 255, 255]

                    if image[int_v - 1][int_u - 1] == 0 and image[int_v - 2][int_u - 2] != 0:
                        image[int_v - 1][int_u - 1] = z1
                        # frameRgb[int_v - 1][int_u - 1] = [255, 255, 255]
                image[int_v][int_u] = z1

                prev_u = int_u
                prev_v_idxs[j] = int_v

    return image

rgbWeight = 0.6
depthWeight = 0.4

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
cam_a = pipeline.create(dai.node.Camera)
tof = pipeline.create(dai.node.ToF)

tofConfig = tof.initialConfig.get()
# tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MIN
tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MAX
tofConfig.depthParams.avgPhaseShuffle = False
tofConfig.depthParams.minimumAmplitude = 3.0
tof.initialConfig.set(tofConfig)
# Link the ToF sensor to the ToF node
cam_a.raw.link(tof.input)

rgbOut = pipeline.create(dai.node.XLinkOut)
xout = pipeline.create(dai.node.XLinkOut)
tof.depth.link(xout.input)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
xout.setStreamName("depth")
queueNames.append("depth")

hardware_rectify = False

fps = 30

RGB_SOCKET = dai.CameraBoardSocket.CAM_C
LEFT_SOCKET = dai.CameraBoardSocket.CAM_C
ToF_SOCKET = dai.CameraBoardSocket.CAM_A

#Properties
camRgb.setBoardSocket(RGB_SOCKET)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setFps(fps)
#camRgb.setIspScale(2, 3)

try:    
    calibData = device.readCalibration2()

    rgb_intrinsics = calibData.getCameraIntrinsics(RGB_SOCKET,640, 480)
    print(f'rgb_intrinsics at 640x 480 {rgb_intrinsics}')
    depth_intrinsics = calibData.getCameraIntrinsics(ToF_SOCKET,640, 480)
    print(f'TOF Intrinsics {depth_intrinsics}')
    distortCoeffs = np.asarray(calibData.getDistortionCoefficients(ToF_SOCKET))
    print(f'TOF Distortion coeffs {distortCoeffs}')
    rgb_extrinsics = calibData.getCameraExtrinsics(ToF_SOCKET, RGB_SOCKET)
    print(f'rgb_extrinsics {np.asarray(rgb_extrinsics)}')
    depth_intrinsics = np.asarray(depth_intrinsics).reshape(3, 3)
    rgb_extrinsics = np.asarray(rgb_extrinsics).reshape(4, 4)
    rgb_extrinsics[:,3] *= 10 # to mm
    rgb_intrinsics = np.asarray(rgb_intrinsics).reshape(3, 3)

    lensPosition = calibData.getLensPosition(RGB_SOCKET)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
# Linking
camRgb.isp.link(rgbOut.input)

def colorizeDepth(frameDepth):
    min_depth = np.percentile(frameDepth[frameDepth != 0], 1)
    max_depth = np.percentile(frameDepth, 99)
    depthFrameColor = np.interp(frameDepth, (min_depth, max_depth), (0, 255)).astype(np.uint8)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
    return depthFrameColor


# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)
    print('Connected cameras:', device.getConnectedCameraFeatures())

    frameRgb = None
    frameDepth = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgb_depth_window_name = 'rgb-depth1'

    cv2.namedWindow(rgb_depth_window_name)
    cv2.createTrackbar('RGB Weight %', rgb_depth_window_name, int(rgbWeight*100), 100, updateBlendWeights)

    rgb_depth_window_name1 = 'rgb-depth2'

    cv2.namedWindow(rgb_depth_window_name1)
    cv2.createTrackbar('RGB Weight %', rgb_depth_window_name1, int(rgbWeight*100), 100, updateBlendWeights)

    depth_intrinsics = np.array([[471.451,   0.   , 317.897],
                                 [  0.   , 471.539, 245.027],
                                 [  0.   ,   0.   ,   1.   ]])
    distortCoeffs = np.array([ 0.081, -0.432,  0.   , -0.   ,  0.418])

    if hardware_rectify:
        rectification_map = cv2.initUndistortRectifyMap(depth_intrinsics, distortCoeffs, rgb_extrinsics[0:3, 0:3], depth_intrinsics, (640, 480), cv2.CV_32FC1)
    else:
        rectification_map = cv2.initUndistortRectifyMap(depth_intrinsics, distortCoeffs, np.eye(3), depth_intrinsics, (640, 480), cv2.CV_32FC1)


    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["depth"] = None

        queueEvents = device.getQueueEvents(("rgb", "depth"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()

        if latestPacket["depth"] is not None:
            frameDepth = latestPacket["depth"].getFrame()
            print(frameDepth.shape)
            depth_downscaled = frameDepth[::4]
            non_zero_depth = depth_downscaled[depth_downscaled != 0] # Remove invalid depth values
            cv2.imshow("ToF", non_zero_depth)
            if len(non_zero_depth) == 0:
                min_depth, max_depth = 0, 0
            else:
                min_depth = np.percentile(non_zero_depth, 3)
                max_depth = np.percentile(non_zero_depth, 97)
            depth_colorized = np.interp(frameDepth, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            frameDepth = depth_colorized
        # Blend when both received
        if frameRgb is not None and frameDepth is not None:

            print(f'Shape of frame depth is {frameDepth.shape}')
            print(f'Shape of rectification map is {rectification_map[0].shape}')
            print(f'Shape of frameRgb is {frameRgb.shape}')
            print('Why rectifiy here ?')
            # exit(1)
            frameDepth = cv2.remap(frameDepth, rectification_map[0], rectification_map[1], cv2.INTER_LINEAR)

            # print(f'Extrinsics {rgb_extrinsics}')
            # print(f'Intrinsics {rgb_intrinsics}')
            # print(f'Depth median is {np.median(frameDepth[frameDepth != 0])}')
            # exit(1)

            # NEEDS RESCALING
            frameDepth1 = reprojection(frameDepth, depth_intrinsics, rgb_extrinsics, rgb_intrinsics, 0, hardware_rectify)
            frameDepth2 = reprojection(frameDepth, depth_intrinsics, rgb_extrinsics, rgb_intrinsics, 1, hardware_rectify)

           
            frameDepth1 = colorizeDepth(frameDepth1)
            frameDepth2 = colorizeDepth(frameDepth2)
            cv2.imshow("Colorized depth", depth_colorized)
            cv2.imshow("Colorized depth1", frameDepth1)
            cv2.imshow("Colorized depth2", frameDepth2)
            
            # frameRgb = cv2.resize(frameRgb, (frameDepth.shape[1], frameDepth.shape[0]))
            frameRgb = cv2.resize(frameRgb, (0, 0), fx=0.5, fy=0.5)
            print('Frame rgb shape -----------------------')
            print(frameRgb.shape)
            frameRgb_padded = np.pad(frameRgb, ((40, 40), (0, 0), (0, 0)), mode='constant', constant_values=0)
            print('frameDepth1 img shape -----------------------')
            print(frameDepth1.shape)

            # exit(1)
            blended = cv2.addWeighted(frameRgb_padded, rgbWeight, frameDepth1, depthWeight, 0)
            cv2.imshow(rgb_depth_window_name, blended)

            blended2 = cv2.addWeighted(frameRgb_padded, rgbWeight, frameDepth2, depthWeight, 0)
            cv2.imshow(rgb_depth_window_name1, blended2)

            frameRgb = None
            frameDepth = None
        if cv2.waitKey(1) == ord('q'):
            break