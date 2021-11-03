# coding=utf-8
from pathlib import Path
import math
import cv2
import depthai as dai
import numpy as np
from palm_detection import PalmDetection

DEPTH_THRESH_HIGH = 3000
DEPTH_THRESH_LOW = 500
WARNING_DIST = 300

PALM_START_X = -1
PALM_START_Y = -1
PALM_START_Z = -1


def crop_to_rect(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]


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

        # Calculate the average depth in the ROI.
        depthROI = depth[ymin:ymax, xmin:xmax]
        inThreshRange = (DEPTH_THRESH_LOW < depthROI) & (depthROI < DEPTH_THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inThreshRange])
        # print(f"Average depth: {averageDepth}")

        # Palm detection centroid
        centroidX = int((xmax - xmin) / 2) + xmin
        centroidY = int((ymax - ymin) / 2) + ymin

        mid = int(depth.shape[0] / 2) # middle of the depth img
        bb_x_pos = centroidX - mid
        bb_y_pos = centroidY - mid

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        # print(f"X: {x}mm, Y: {y} mm, Z: {z} mm")
        return (x,y,z, centroidX, centroidY)

    def calc_spatial_distance(self, spatialCoords, frame, spatialCoords_fixed_3dPoint):
        x,y,z, centroidX, centroidY = spatialCoords
        fixed_x, fixed_y, fixed_z = spatialCoords_fixed_3dPoint

        annotate = annotate_fun(frame, (0, 0, 25), fontScale=1.7)
        
        self.distance = math.sqrt((fixed_x-x)**2 + (fixed_y-y)**2 + (fixed_z-z)**2)

        height = frame.shape[0]
        cv2.line(frame, (centroidX, centroidY), (int(fixed_x), int(fixed_y)), (50,220,100), 4)

        if self.distance < WARNING_DIST:
            annotate("Danger", (100, int(height/3)))
            annotate("Danger", (101, int(height/3)))
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

    def calc_angle(self, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (self.depthWidth / 2.0))

    def draw_palm_detection(self, palm_coords, depth):
        if palm_coords is None:
            return None

        color = (10,245,10)
        annotate = annotate_fun(self.debug_frame, color)

        for bbox in palm_coords:
            self.draw_bbox(bbox, color)
            spatialCoords = self.calc_spatials(bbox, depth)
            x,y,z,cx,cy = spatialCoords
            if math.isnan(x) | math.isnan(y) | math.isnan(z):
                return None
            annotate(f"X: {int(x)} mm", (bbox[0], bbox[1]))
            annotate(f"Y: {int(y)} mm", (bbox[0], bbox[1] + 15))
            annotate(f"Z: {int(z)} mm", (bbox[0], bbox[1] + 30))
            return spatialCoords
    
    def generate_fixed_3DPoint(self, spatialCoords):
        # TODO: set to initial detected point of pose
        x,y,z, centroidX, centroidY = spatialCoords
        
        # ignore the detected bottle and use a fixed point
        global PALM_START_X
        global PALM_START_Y
        global PALM_START_Z
        if PALM_START_X == -1:
            PALM_START_X = 127
            PALM_START_Y = 114
            PALM_START_Z = 1082
        return (PALM_START_X, PALM_START_Y, PALM_START_Z)
    
    def parse(self, in_palm_detections, frame, depth, depthColored):
        self.debug_frame = frame.copy()
        self.depthFrameColor = depthColored
        annotate = annotate_fun(self.debug_frame, (50,220,110), fontScale=1.4)

        # Parse palm detection output
        palm_coords = self.palmDetection.run_palm(
            self.debug_frame,
            in_palm_detections)
        # Calculate and draw spatial coordinates of the palm
        spatialCoords = self.draw_palm_detection(palm_coords, depth)

        # 3D_point in space detection
        if spatialCoords is not None:
            spatialCoords_fixed_3dPoint = self.generate_fixed_3DPoint(spatialCoords)
        
        # Calculate distance, show warning
        if spatialCoords is not None:
            self.calc_spatial_distance(spatialCoords,self.debug_frame, spatialCoords_fixed_3dPoint)

        # Put text 3 times for the bold appearance
        if math.isnan(self.distance):
            return None
        annotate(f"Distance: {int(self.distance)} mm", (50,700))
        annotate(f"Distance: {int(self.distance)} mm", (51,700))
        annotate(f"Distance: {int(self.distance)} mm", (52,700))
        cv2.imshow("color", self.debug_frame)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration()


print("Creating pipeline...")
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(2, 3) # To match 720P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)
# For MobileNet NN
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)

isp_xout = pipeline.createXLinkOut()
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

print(f"Creating palm detection Neural Network...")
model_nn = pipeline.createNeuralNetwork()
model_nn.setBlobPath(str(Path("models/palm_detection_openvino_2021.3_6shave.blob").resolve().absolute()))
model_nn.input.setBlocking(False)

# For Palm-detection NN
manip = pipeline.createImageManip()
manip.initialConfig.setResize(128, 128)
cam.preview.link(manip.inputImage)
manip.out.link(model_nn.input)

model_nn_xout = pipeline.createXLinkOut()
model_nn_xout.setStreamName("palm_nn")
model_nn.out.link(model_nn_xout.input)

# Creating left/right mono cameras for StereoDepth
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create StereoDepth node that will produce the depth map
stereo = pipeline.createStereoDepth()
stereo.initialConfig.setConfidenceThreshold(245)
stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

sdn = pipeline.createMobileNetSpatialDetectionNetwork()
sdn.setBlobPath("models/mobilenet-ssd_openvino_2021.2_6shave.blob")
sdn.setConfidenceThreshold(0.5)
sdn.input.setBlocking(False)
sdn.setBoundingBoxScaleFactor(0.2)
sdn.setDepthLowerThreshold(DEPTH_THRESH_LOW)
sdn.setDepthUpperThreshold(DEPTH_THRESH_HIGH)

cam.preview.link(sdn.input)
stereo.depth.link(sdn.inputDepth)

sdn_out = pipeline.createXLinkOut()
sdn_out.setStreamName("det")
sdn.out.link(sdn_out.input)

depth_out = pipeline.createXLinkOut()
depth_out.setStreamName("depth")
sdn.passthroughDepth.link(depth_out.input)

print("Pipeline created.")

with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(pipeline)
    # Create output queues
    vidQ = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    detQ = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    palmQ = device.getOutputQueue(name="palm_nn", maxSize=4, blocking=False)

    humanMachineSafety = HumanMachineSafety()

    depthFrame = None
    depthFrameColor = None
    frame = None

    while True:
        in_rgb = vidQ.tryGet()
        if in_rgb is not None:
            frame = crop_to_rect(in_rgb.getCvFrame())

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
                    frame,
                    depthFrame,
                    depthFrameColor)
            except StopIteration:
                break
