# coding=utf-8
import math

import blobconverter
import cv2
import depthai as dai
import numpy as np

from palm_detection import PalmDetection

DEPTH_THRESH_HIGH = 3000
DEPTH_THRESH_LOW = 500
WARNING_DIST = 300

# If dangerous object is too close to the palm, warning will be displayed
DANGEROUS_OBJECTS = ["bottle"]

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def crop_to_rect(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width - delta]


def annotate_fun(img, color, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, **kwargs):
    def fun(text, pos):
        cv2.putText(img, text, pos, fontFace, fontScale, color, **kwargs)

    return fun


class HumanMachineSafety:
    def __init__(self):
        print("Loading pipeline...")
        self.palmDetection = PalmDetection()

        self.distance = 1000

        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(73.5)
        self.depthWidth = 1080.0

    # Calculate spatial coordinates from depth map and bounding box (ROI)
    def calc_spatials(self, bbox, depth, averaging_method=np.mean):
        xmin, ymin, xmax, ymax = bbox
        # Decrese the ROI to 1/3 of the original ROI
        deltaX = int((xmax - xmin) * 0.33)
        deltaY = int((ymax - ymin) * 0.33)
        xmin += deltaX
        ymin += deltaY
        xmax -= deltaX
        ymax -= deltaY
        if xmin > xmax:  # bbox flipped
            xmin, xmax = xmax, xmin
        if ymin > ymax:  # bbox flipped
            ymin, ymax = ymax, ymin

        if xmin == xmax or ymin == ymax:  # Box of size zero
            return None

        # Calculate the average depth in the ROI.
        depthROI = depth[ymin:ymax, xmin:xmax]
        inThreshRange = (DEPTH_THRESH_LOW < depthROI) & (depthROI < DEPTH_THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inThreshRange])
        # print(f"Average depth: {averageDepth}")

        # Palm detection centroid
        centroidX = int((xmax - xmin) / 2) + xmin
        centroidY = int((ymax - ymin) / 2) + ymin

        mid = int(depth.shape[0] / 2)  # middle of the depth img
        bb_x_pos = centroidX - mid
        bb_y_pos = centroidY - mid

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        # print(f"X: {x}mm, Y: {y} mm, Z: {z} mm")
        return (x, y, z, centroidX, centroidY)

    def calc_spatial_distance(self, spatialCoords, frame, detections):
        x, y, z, centroidX, centroidY = spatialCoords
        annotate = annotate_fun(frame, (0, 0, 25), fontScale=1.7)

        for det in detections:
            # Ignore detections that aren't considered dangerous
            if labelMap[det.label] not in DANGEROUS_OBJECTS: continue

            self.distance = math.sqrt((det.spatialCoordinates.x - x) ** 2 + (det.spatialCoordinates.y - y) ** 2 + (
                        det.spatialCoordinates.z - z) ** 2)

            height = frame.shape[0]
            x1 = int(det.xmin * height)
            x2 = int(det.xmax * height)
            y1 = int(det.ymin * height)
            y2 = int(det.ymax * height)
            objectCenterX = int((x1 + x2) / 2)
            objectCenterY = int((y1 + y2) / 2)
            cv2.line(frame, (centroidX, centroidY), (objectCenterX, objectCenterY), (50, 220, 100), 4)

            if self.distance < WARNING_DIST:
                # Color dangerous objects in red
                sub_img = frame[y1:y2, x1:x2]
                red_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                # Setting blue/green to 0
                red_rect[:, :, 0] = 0
                red_rect[:, :, 1] = 0
                res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                # Putting the image back to its position
                frame[y1:y2, x1:x2] = res
                # Print twice to appear in bold
                annotate("Danger", (100, int(height / 3)))
                annotate("Danger", (101, int(height / 3)))
        cv2.imshow("color", frame)

    def draw_bbox(self, bbox, color):
        def draw(img):
            cv2.rectangle(
                img=img,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=color,
                thickness=2,
            )

        draw(self.debug_frame)
        draw(self.depthFrameColor)

    def draw_detections(self, frame, detections):
        color = (250, 0, 0)
        annotate = annotate_fun(frame, (0, 0, 25))

        for detection in detections:
            if labelMap[detection.label] not in DANGEROUS_OBJECTS: continue
            height = frame.shape[0]
            width = frame.shape[1]
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            offsetX = x1 + 10
            annotate("{:.2f}".format(detection.confidence * 100), (offsetX, y1 + 35))
            annotate(f"X: {int(detection.spatialCoordinates.x)} mm", (offsetX, y1 + 50))
            annotate(f"Y: {int(detection.spatialCoordinates.y)} mm", (offsetX, y1 + 65))
            annotate(f"Z: {int(detection.spatialCoordinates.z)} mm", (offsetX, y1 + 80))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            annotate(str(label), (offsetX, y1 + 20))

    def calc_angle(self, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (self.depthWidth / 2.0))

    def draw_palm_detection(self, palm_coords, depth):
        if palm_coords is None:
            return None

        color = (10, 245, 10)
        annotate = annotate_fun(self.debug_frame, color)

        for bbox in palm_coords:
            spatialCoords = self.calc_spatials(bbox, depth)
            if spatialCoords is None:  # Box of size 0
                continue
            self.draw_bbox(bbox, color)
            x, y, z, cx, cy = spatialCoords
            annotate(f"X: {int(x)} mm", (bbox[0], bbox[1]))
            annotate(f"Y: {int(y)} mm", (bbox[0], bbox[1] + 15))
            annotate(f"Z: {int(z)} mm", (bbox[0], bbox[1] + 30))
            return spatialCoords

    def parse(self, in_palm_detections, detections, frame, depth, depthColored):
        self.debug_frame = frame.copy()
        self.depthFrameColor = depthColored
        annotate = annotate_fun(self.debug_frame, (50, 220, 110), fontScale=1.4)

        # Parse palm detection output
        palm_coords = self.palmDetection.run_palm(
            self.debug_frame,
            in_palm_detections)
        # Calculate and draw spatial coordinates of the palm
        spatialCoords = self.draw_palm_detection(palm_coords, depth)
        # Calculate distance, show warning
        if spatialCoords is not None:
            self.calc_spatial_distance(spatialCoords, self.debug_frame, detections)

        # Mobilenet detections
        self.draw_detections(self.debug_frame, detections)
        # Put text 3 times for the bold appearance
        annotate(f"Distance: {int(self.distance)} mm", (50, 700))
        annotate(f"Distance: {int(self.distance)} mm", (51, 700))
        annotate(f"Distance: {int(self.distance)} mm", (52, 700))
        cv2.imshow("color", self.debug_frame)

        if self.depthFrameColor is not None:
            self.draw_detections(self.depthFrameColor, detections)
            cv2.imshow("depth", self.depthFrameColor)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration()


print("Creating pipeline...")
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(2, 3)  # To match 720P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)
# For MobileNet NN
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)

isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

print(f"Creating palm detection Neural Network...")
model_nn = pipeline.create(dai.node.NeuralNetwork)
model_nn.setBlobPath(blobconverter.from_zoo(name="palm_detection_128x128", zoo_type="depthai", shaves=6))
model_nn.input.setBlocking(False)

# For Palm-detection NN
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(128, 128)
cam.preview.link(manip.inputImage)
manip.out.link(model_nn.input)

model_nn_xout = pipeline.create(dai.node.XLinkOut)
model_nn_xout.setStreamName("palm_nn")
model_nn.out.link(model_nn_xout.input)

# Creating left/right mono cameras for StereoDepth
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create StereoDepth node that will produce the depth map
stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(245)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

sdn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
sdn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
sdn.setConfidenceThreshold(0.5)
sdn.input.setBlocking(False)
sdn.setBoundingBoxScaleFactor(0.2)
sdn.setDepthLowerThreshold(DEPTH_THRESH_LOW)
sdn.setDepthUpperThreshold(DEPTH_THRESH_HIGH)

cam.preview.link(sdn.input)
stereo.depth.link(sdn.inputDepth)

sdn_out = pipeline.create(dai.node.XLinkOut)
sdn_out.setStreamName("det")
sdn.out.link(sdn_out.input)

depth_out = pipeline.create(dai.node.XLinkOut)
depth_out.setStreamName("depth")
sdn.passthroughDepth.link(depth_out.input)

print("Pipeline created.")

with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError(
            "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(pipeline)
    # Create output queues
    vidQ = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    palmQ = device.getOutputQueue(name="palm_nn", maxSize=4, blocking=False)

    humanMachineSafety = HumanMachineSafety()

    detections = []
    depthFrame = None
    depthFrameColor = None
    frame = None

    while True:
        in_rgb = vidQ.tryGet()
        if in_rgb is not None:
            frame = crop_to_rect(in_rgb.getCvFrame())

        # Check for mobilenet detections
        in_det = detQ.tryGet()
        if in_det is not None:
            detections = in_det.detections

        in_depth = depthQ.tryGet()
        if in_depth is not None:
            depthFrame = crop_to_rect(in_depth.getFrame())
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        palm_in = palmQ.tryGet()
        if palm_in is not None and frame is not None and depthFrame is not None:
            try:
                humanMachineSafety.parse(
                    palm_in,
                    detections,
                    frame,
                    depthFrame,
                    depthFrameColor)
            except StopIteration:
                break
