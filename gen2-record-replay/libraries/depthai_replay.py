from pathlib import Path
import os
import cv2
import types
import depthai as dai
import numpy as np
import json
import math

def getMesh(calibData, resolution, offset):
    print("------mesh res", resolution, "offset", offset) # TODO see if offset is needed here and implement...
    topLeftPixel = dai.Point2f(offset[0], resolution[1])
    bottomRightPixel = dai.Point2f(resolution[0], 0)

    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1], topLeftPixel, bottomRightPixel))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1], topLeftPixel, bottomRightPixel))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    tranformation = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    R = tranformation[:3, :3]
    T = tranformation[:3, 3]

    debug = False
    if debug: 
        print('printing transformation matrix')
        print(tranformation)

        print(R)
        print('printing Tranlsation vec')

        print(T)
        print('Printing old R1 and R2')
        print(R1)
        print(R2)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                                                                M1,
                                                                d1,
                                                                M2,
                                                                d2,
                                                                resolution, R, T)

    print('Printing new R1 and R2')
    print(R1)
    print(R2)
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, P1, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, P2, resolution, cv2.CV_32FC1)

    meshCellSize = 16
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

class Replay:
    def __init__(self, path, lrMode, subpixelMode, extendedMode, offset):
        self.path = Path(path).resolve().absolute()

        self.lrMode = lrMode
        self.subpixelMode = subpixelMode
        self.extendedMode = extendedMode
        self.cap = {} # VideoCapture objects
        self.size = {} # Frame sizes
        self.offset = {} # Offset from top-left corner
        self.lastFrame = {} # Last frame sent to the device
        self.frames = {} # Frames read from the VideoCapture
        # Disparity shouldn't get streamed to the device, nothing to do with it.
        self.stream_types = ['color', 'left', 'right', 'depth']

        file_types = ['color', 'left', 'right', 'disparity', 'depth']
        extensions = ['mjpeg', 'avi', 'mp4', 'h265', 'h264']

        for file in os.listdir(path):
            if not '.' in file: continue # Folder
            name, extension = file.split('.')
            if name in file_types and extension in extensions:
                self.cap[name] = cv2.VideoCapture(str(self.path / file))

        if len(self.cap) == 0:
            raise RuntimeError("There are no recordings in the folder specified.")

        # Load calibration data from the recording folder
        self.calibData = dai.CalibrationHandler(str(self.path / "calib.json"))
        # StereoDepth hardware limitation
        self.maxWidth = 1280
        # Read basic info about the straems (resolution of streams etc.)
        for name in self.cap:
            self.size[name] = self.get_size(self.cap[name])
            self.offset[name] = (0, 0)
            # TODO maybe move the cropped size info elsewhere
            if name in ['left', 'right'] and self.size[name][0] > self.maxWidth:
                print(f'Cropping {name:5}: {self.size[name]} -> ', end='')
                self.size[name] = (min(self.size[name][0], self.maxWidth), self.size[name][1])
                self.offset[name] = (offset, 0)
                print(f'{self.size[name]}, offset {self.offset[name]}')

        self.color_size = None
        # By default crop image as needed to keep the aspect ratio
        self.keep_ar = True

    # Resize color frames prior to sending them to the device
    def set_resize_color(self, size):
        self.color_size = size
    def keep_aspect_ratio(self, keep_aspect_ratio):
        self.keep_ar = keep_aspect_ratio

    def disable_stream(self, stream_name):
        if stream_name not in self.cap:
            print(f"There's no stream {stream_name} available!")
            return
        self.cap[stream_name].release()
        # Remove the stream from the VideoCapture dict
        self.cap.pop(stream_name, None)

    def resize_color(self, frame):
        if self.color_size is None:
            # No resizing needed
            return frame

        if not self.keep_ar:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self.color_size)

        h = frame.shape[0]
        w = frame.shape[1]
        desired_ratio = self.color_size[0] / self.color_size[1]
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if desired_ratio < current_ratio: # Crop width
            # Use full height, crop width
            new_w = (desired_ratio/current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop:w-crop]
        else: # Crop height
            # Use full width, crop height
            new_h = (current_ratio/desired_ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop:h-crop,:]

        return cv2.resize(preview, self.color_size)

    def init_pipeline(self):
        nodes = {}
        mono = 'left' and 'right' in self.cap
        depth = 'depth' in self.cap
        if mono and depth: # This should be possible either way.
            mono = False # Use depth stream by default

        pipeline = dai.Pipeline()
        pipeline.setCalibrationData(self.calibData)
        nodes = types.SimpleNamespace()

        if 'color' in self.cap:
            nodes.color = pipeline.create(dai.node.XLinkIn)
            nodes.color.setMaxDataSize(self.get_max_size('color'))
            nodes.color.setStreamName("color_in")

        if mono:
            nodes.left = pipeline.create(dai.node.XLinkIn)
            nodes.left.setStreamName("left_in")
            nodes.left.setMaxDataSize(self.get_max_size('left'))

            nodes.right = pipeline.create(dai.node.XLinkIn)
            nodes.right.setStreamName("right_in")
            nodes.right.setMaxDataSize(self.get_max_size('right'))

            nodes.stereo = pipeline.create(dai.node.StereoDepth)
            nodes.stereo.setInputResolution(self.size['left'][0], self.size['left'][1])
            nodes.stereo.setLeftRightCheck(self.lrMode)
            nodes.stereo.setSubpixel(self.subpixelMode)
            nodes.stereo.setExtendedDisparity(self.extendedMode)
            nodes.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            # black instead of pixel replication for missing frame data at the edges
            nodes.stereo.setRectifyEdgeFillColor(0)
            self.meshLeft, self.meshRight = getMesh(self.calibData, self.size['left'], self.offset['left'])
            meshLeft = list(self.meshLeft.tobytes())
            meshRight = list(self.meshRight.tobytes())

            nodes.stereo.loadMeshData(meshLeft, meshRight)
            # nodes.stereoSetFocalLengthFromCalibration(True)
            nodes.left.out.link(nodes.stereo.left)
            nodes.right.out.link(nodes.stereo.right)
            
        if depth:
            nodes.depth = pipeline.create(dai.node.XLinkIn)
            nodes.depth.setStreamName("depth_in")

        # TODO use calibration handler instead
        with open(self.path / "calib.json", 'r') as f:
            j = json.load(f)
            tCal = j['cameraData'][0][1]['extrinsics']['translation']
            baselineSpec = abs(j['cameraData'][0][1]['extrinsics']['specTranslation']['x'])
            baselineCalib = math.sqrt(pow(tCal['x'], 2) + pow(tCal['y'], 2) + pow(tCal['z'], 2))
            print(f'Baseline spec: {baselineSpec:.3f}, calib: {baselineCalib:.3f} cm')
            focalRightCalib = abs(j['cameraData'][1][1]['intrinsicMatrix'][0][0])
            #print(f'Focal right calib: {focalRightCalib:.3f}')
            self.disp_levels = 1
            if self.subpixelMode:
                subpixelBits = nodes.stereo.initialConfig.get().algorithmControl.subpixelFractionalBits
                self.disp_levels = pow(2, subpixelBits)
            factor = focalRightCalib * self.disp_levels
            self.disp_to_depth_factor_spec_baseline  = factor * (baselineSpec  / 100)
            self.disp_to_depth_factor_calib_baseline = factor * (baselineCalib / 100)

        return pipeline, nodes

    def create_queues(self, device):
        self.queues = {}
        for name in self.cap:
            if name in self.stream_types:
                self.queues[name+'_in'] = device.getInputQueue(name+'_in')

    def to_planar(self, arr, shape = None):
        if shape is not None: arr = cv2.resize(arr, shape)
        return arr.transpose(2, 0, 1).flatten()

    def read_frames(self):
        self.frames = {}
        for name in self.cap:
            if not self.cap[name].isOpened():
                return True
            ok, frame = self.cap[name].read()
            if ok:
                if name in ['left', 'right']:
                    # Crop input frame for StereoDepth constraints
                    x, y = self.offset[name]
                    w, h = self.size[name]
                    frame = frame[y:y+h, x:x+w]
                self.frames[name] = frame
        return len(self.frames) == 0

    def send_frames(self):
        if self.read_frames():
            return False # end of recording
        for name in self.frames:
            if name in ["left", "right", "disparity"] and len(self.frames[name].shape) == 3:
                self.frames[name] = self.frames[name][:,:,0] # All 3 planes are the same

            self.send_frame(self.frames[name], name)

        return True

    def get_size(self, cap):
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    def get_max_size(self, name):
        total = self.size[name][0] * self.size[name][1]
        if name == 'color': total *= 3 # 3 channels
        return total

    def send_frame(self, frame, name):
        q_name = name + '_in'
        if q_name in self.queues:
            if name == 'color':
                self.send_color(self.queues[q_name], frame)
            elif name == 'left':
                self.send_mono(self.queues[q_name], frame, False)
            elif name == 'right':
                self.send_mono(self.queues[q_name], frame, True)
            elif name == 'depth':
                self.send_depth(self.queues[q_name], frame)

    def send_mono(self, q, img, right):
        self.lastFrame['right' if right else 'left'] = img
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(img)
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if right else 1))
        q.send(frame)

    def send_color(self, q, img):
        # Resize/crop color frame as specified by the user
        img = self.resize_color(img)
        self.lastFrame['color'] = img
        h, w, c = img.shape
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(img))
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum(0)
        q.send(frame)

    def send_depth(self, q, depth):
        # TODO refactor saving depth. Reading will be from ROS bags.

        # print("depth size", type(depth))
        # depth_frame = np.array(depth).astype(np.uint8).view(np.uint16).reshape((400, 640))
        # depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        # cv2.imshow("depth", depthFrameColor)
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        q.send(frame)