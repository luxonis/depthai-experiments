#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import math
import time
import depthai as dai
from crash_avoidance import CrashAvoidance

IMG_SAVE_PATH = "/frames"

for device in dai.Device.getAllAvailableDevices():
    print(f"{device.getMxId()} {device.state}")

labelMap = ["background", "vehicle"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--depth', action='store_true', default=False, help="Use saved depth maps")
parser.add_argument('-m', '--monos', action='store_true', default=False, help="Display synced mono frames as well")
parser.add_argument('-mh', '--monohost', action='store_true', default=False, help="Display  mono frames from the host")
parser.add_argument('-t', '--tracker', action='store_true', default=False, help="Use object tracker")
parser.add_argument('-bv', '--birdsview', action='store_true', default=False, help="Show birds view")
parser.add_argument('-mx', '--mx', default="", type=str, help="MX id")
args = parser.parse_args()

model_width = 672
model_height = 384
MIN_Z = 2000
MAX_Z = 35000
model_path = "models/vehicle-detection-adas-0002.blob"

# Get the stored frames path
dest = Path(args.path).resolve().absolute()
frame_folders = os.listdir(str(dest))

frames_sorted = []
for frame_folder in frame_folders:
    try:
        frames_sorted.append(int(frame_folder))
    except Exception as e:
        print(f"Folder named {frame_folder} is invalid! Skipping it")
frames_sorted = sorted(frames_sorted)

# Custom JET colormap with 0 mapped to `black` - better disparity visualization
jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom[0] = [0, 0, 0]

class Replay:
    def __init__(self, path):
        self.path = path
        self.mono_size = self.__get_mono_size()
        self.color_size = self.__get_color_size()

    def set_device(self, device):
        self.device = device
    # Crop & resize the frame (with the correct aspect ratio)
    def crop_frame(self, frame):
        shape = frame.shape
        h = shape[0]
        w = shape[1]
        ratio = model_width / model_height
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if ratio < current_ratio: # Crop width
            # Use full height, crop width
            new_w = (ratio/current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop:w-crop]
        else: # Crop height
            # Use full width, crop height
            new_h = (current_ratio/ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop:h-crop,:]

        # Resize to match the NN input
        return preview

    def create_input_queues(self):
        # Create input queues
        inputs = ["rgbIn"]
        if args.depth:
            inputs.append("depthIn")
        else: # Use mono frames
            inputs.append("left")
            inputs.append("right")
        self.q = {}
        for input_name in inputs:
            self.q[input_name] = self.device.getInputQueue(input_name)

    def __get_color_size(self):
        files = self.get_files(1000)
        for file in files:
            if not file.startswith("color"): continue
            frame = self.read_color(self.get_path(1000, file))
            return frame.shape
        return None

    def __get_mono_size(self):
        files = self.get_files(1000)
        for file in files:
            if not file.startswith("left"): continue
            frame = self.read_mono(self.get_path(1000, file))
            return frame.shape
        return None

    def __to_planar(self, arr, shape):
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def read_color(self,path):
        return cv2.imread(path)
    def read_mono(self,path):
        return cv2.flip(cv2.imread(path, cv2.IMREAD_GRAYSCALE), 1)
    def read_depth(self,path):
        with open(path, "rb") as depth_file:
            return list(depth_file.read())
    def get_path(self, folder, file = None):
        if file is None:
            return str((Path(self.path) / str(folder)).resolve().absolute())
        return str((Path(self.path) / str(folder) / file).resolve().absolute())

    def read_files(self, frame_folder, files):
        frames = {}
        for file in files:
            file_path = self.get_path(frame_folder, file)
            if file.startswith("right") or file.startswith("left"):
                frame = self.read_mono(file_path)
            elif file.startswith("depth"):
                frame = self.read_depth(file_path)
            elif file.startswith("color"):
                frame = self.read_color(file_path)
            else:
                # print(f"Unknown file found! {file}")
                continue
            frames[os.path.splitext(file)[0]] = frame
        return frames

    def get_files(self, frame_folder):
        return os.listdir(self.get_path(frame_folder))

    def send_frames(self, images):
        for name, img in images.items():
            replay.send_frame(name, img)

    # Send recorded frames from the host to the depthai
    def send_frame(self, name, frame):
        if name in ["left", "right"] and not args.depth:
            self.send_mono(frame, name)
        elif name == "depth" and args.depth:
            self.send_depth(frame)
        elif name == "color":
            self.send_rgb(frame)

    def send_mono(self, img, name):
        frame = dai.ImgFrame()
        frame.setData(img) # Flip the rectified frame
        frame.setType(dai.RawImgFrame.Type.RAW8)
        h, w = img.shape
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if name == "right" else 1))
        self.q[name].send(frame)
    def send_rgb(self, img):
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        preview = self.crop_frame(img)
        frame.setData(self.__to_planar(preview, (model_width, model_height)))
        frame.setWidth(model_width)
        frame.setHeight(model_height)
        frame.setInstanceNum(0)
        self.q["rgbIn"].send(frame)
    def send_depth(self, depth):
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        self.q["depthIn"].send(frame)

def create_bird_frame():
    fov = 68.3
    frame = np.zeros((300, 100, 3), np.uint8)
    cv2.rectangle(frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
    return frame

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.4, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.4, self.color, 1, self.line_type)

textHelper = TextHelper()

def draw_bird_frame(frame, x, z, id = None):
    global MAX_Z
    max_x = 15000 #mm
    pointY = frame.shape[0] - int(z / (MAX_Z - 10000) * frame.shape[0]) - 20
    pointX = int(x / max_x * frame.shape[1] + frame.shape[1]/2)
    # print(f"Y {y}, Z {z} - Birds: X {pointX}, Y {pointY}")
    if id is not None:
        cv2.putText(frame, str(id), (pointX - 30, pointY + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
    cv2.circle(frame, (pointX, pointY), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)

crashAvoidance = CrashAvoidance()
# Draw spatial detections / tracklets to the frame
def display_spatials(frame, detections, name, tracker = False):
    color = (186, 186, 186) if name == "depth" else (30, 211, 255)
    h = frame.shape[0]
    w = frame.shape[1]
    birdFrame = create_bird_frame()
    for detection in detections:
        # Denormalize bounding box
        imgDet = detection.srcImgDetection if tracker else detection
        x1 = int(imgDet.xmin * w)
        x2 = int(imgDet.xmax * w)
        y1 = int(imgDet.ymin * h)
        y2 = int(imgDet.ymax * h)

        if tracker:
            if crashAvoidance.remove_lost_tracklet(detection): continue
            # If these are tracklets, display ID as well
            textHelper.putText(frame, f"Car {detection.id}", (x1 + 10, y1 + 20))
            # frame = crashAvoidance.drawArrows(frame, detection)
            # cv2.putText(frame, detection.status.name, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # cv2.putText(frame, "{:.2f}".format(detection.srcImgDetection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # speed = crashAvoidance.calculate_speed(detection)

            # cv2.putText(frame, "{:.1f} km/h".format(speed), (x1 + 10, y1 - 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # else:
            # cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (230,230,230), cv2.FONT_HERSHEY_SIMPLEX)
        # If vehicle is too far away, coordinate estimation is off so don't display them
        if (x2-x1)*(y2-y1) < 600: continue
        draw_bird_frame(birdFrame, detection.spatialCoordinates.x, detection.spatialCoordinates.z, detection.id if tracker else None)
        textHelper.putText(frame, "X: {:.1f} m".format(int(detection.spatialCoordinates.x) / 1000.0), (x1 + 10, y1 + 35))
        textHelper.putText(frame, "Y: {:.1f} m".format(int(detection.spatialCoordinates.y) / 1000.0), (x1 + 10, y1 + 50))
        textHelper.putText(frame, "Z: {:.1f} m".format(int(detection.spatialCoordinates.z) / 1000.0), (x1 + 10, y1 + 65))
    if args.birdsview: return birdFrame

replay = Replay(path=args.path)

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

rgb_in = pipeline.createXLinkIn()
rgb_in.setStreamName("rgbIn")

if not args.depth:
    left_in = pipeline.createXLinkIn()
    right_in = pipeline.createXLinkIn()
    left_in.setStreamName("left")
    right_in.setStreamName("right")

    stereo = pipeline.createStereoDepth()
    stereo.setConfidenceThreshold(240)
    median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    stereo.setMedianFilter(median)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    stereo.setLeftRightCheck(True)
    mono_size = replay.mono_size
    stereo.setInputResolution(mono_size[1], mono_size[0])
    # Since frames are already rectified
    stereo.setEmptyCalibration()

    left_in.out.link(stereo.left)
    right_in.out.link(stereo.right)
    if args.monos:
        right_s_out = pipeline.createXLinkOut()
        right_s_out.setStreamName("rightS")
        stereo.syncedRight.link(right_s_out.input)

        left_s_out = pipeline.createXLinkOut()
        left_s_out.setStreamName("leftS")
        stereo.syncedLeft.link(left_s_out.input)

spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
spatialDetectionNetwork.setBlobPath(model_path)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(MIN_Z)
spatialDetectionNetwork.setDepthUpperThreshold(MAX_Z)

if args.depth:
    depth_in = pipeline.createXLinkIn()
    depth_in.setStreamName("depthIn")
    depth_in.out.link(spatialDetectionNetwork.inputDepth)
else:
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

rgb_in.out.link(spatialDetectionNetwork.input)

depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("depth")
stereo.depth.link(depthOut.input)

rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("rgb")
spatialDetectionNetwork.passthrough.link(rgbOut.input)

detOut = pipeline.createXLinkOut()
detOut.setStreamName("det")
if args.tracker:
    tracker = pipeline.createObjectTracker()

    tracker.setDetectionLabelsToTrack(range(20))  # track only vehicle
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    tracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)

    spatialDetectionNetwork.passthrough.link(tracker.inputTrackerFrame)
    spatialDetectionNetwork.passthrough.link(tracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(tracker.inputDetections)

    trackerOut = pipeline.createXLinkOut()
    trackerOut.setStreamName("tracklets")
    tracker.out.link(detOut.input)

    # tracker.passthroughDetections.link(detOut.input)
else:
    spatialDetectionNetwork.out.link(detOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    replay.set_device(device)
    replay.create_input_queues()

    if not args.depth and args.monos:
        qLeftS = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
        qRightS = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)

    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    qDet = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    qRgbOut = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    disp_multiplier = 255 / stereo.getMaxDisparity()

    # if args.tracker:
    #     qTracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    for frame_folder in frames_sorted:
        print(frame_folder)
        files = replay.get_files(frame_folder)

        # Read the frames from the FS
        images = replay.read_files(frame_folder, files)

        replay.send_frames(images)

        inRgb = qRgbOut.tryGet()
        if inRgb is not None:
            rgbFrame = inRgb.getCvFrame().reshape((model_height, model_width, 3))

            def get_colored_depth(frame):
                frame = replay.crop_frame(frame)
                depthFrameColor = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                return cv2.applyColorMap(depthFrameColor, jet_custom)

            depthFrameColor = get_colored_depth(qDepth.get().getFrame())

            if args.tracker: detections = qDet.get().tracklets
            else: detections = qDet.get().detections

            birdsView = display_spatials(rgbFrame, detections, "color", args.tracker)
            def save_png(folder, name, item):
                frames_path = Path(IMG_SAVE_PATH) / str(name)
                frames_path.mkdir(parents=True, exist_ok=True)
                if folder < 10: folder = "000" + str(folder)
                elif folder < 100: folder = "00" + str(folder)
                elif folder < 1000: folder = "0" + str(folder)
                cv2.imwrite(str(frames_path / f"{folder}.png"), item)

            h = rgbFrame.shape[0]
            w = rgbFrame.shape[1]
            if not args.depth and args.monos:
                leftS = qLeftS.get().getCvFrame()
                rightS = qRightS.get().getCvFrame()
                left = cv2.resize(leftS, (w,h))
                right = cv2.resize(rightS, (w,h))
                cv2.imshow("left", left)
                cv2.imshow("right", right)
                save_png(frame_folder, "left", left)
                save_png(frame_folder, "right", right)
            if args.monohost:
                cv2.imshow("left", cv2.resize(images["left"], (w,h)))
                cv2.imshow("right", cv2.resize(images["right"], (w,h)))

            cv2.imshow("rgb", rgbFrame)
            depthFrameColor = cv2.resize(depthFrameColor, (w,h))
            display_spatials(depthFrameColor, detections, "depth", args.tracker)
            cv2.imshow("depth", depthFrameColor)
            save_png(frame_folder, "rgb", rgbFrame)
            save_png(frame_folder, "depth", depthFrameColor)
            cv2.imshow("birdsView", birdsView)
            save_png(frame_folder, "birdsview", birdsView)


        if cv2.waitKey(1) == ord('q'):
            break

