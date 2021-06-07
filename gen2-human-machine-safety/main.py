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

LABEL_WARNING = ["bottle"]

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class HumanMachineSafety:
    def __init__(self):
        print("Loading pipeline...")
        self.fontScale = 1
        self.lineType = 0
        self.palmDetection = PalmDetection()

        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(73.5)
        self.depthWidth = 1080.0


    def create_queues(self, device):
        self.vidQ = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
        self.detQ = device.getOutputQueue(name="det", maxSize=4, blocking=False)
        self.depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.palm_nn = device.getOutputQueue(name="palm_nn", maxSize=4, blocking=False)

    def run(self):
        self.depthFrameColor = None
        self.detections = []
        self.depth = None
        self.distance = 1000

        while True:
            in_rgb = self.vidQ.tryGet()
            if in_rgb is not None:
                self.frame = in_rgb.getCvFrame()
                # cv2.imshow("video", self.frame)

                # Check for mobilenet detections
                in_det = self.detQ.tryGet()
                if in_det is not None:
                    self.detections = in_det.detections

                self.frame = self.crop_to_rect(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break


            in_depth = self.depthQ.tryGet()
            if in_depth is not None:
                self.depth = in_depth.getFrame()
                depthFrameColor = cv2.normalize(self.depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                self.depthFrameColor = self.crop_to_rect(depthFrameColor)

    # Calculate spatial coordinates from depth map and bounding box (ROI)
    def calc_spatials(self, bbox, depth):
        croppedDepth = self.crop_to_rect(depth)
        # Decrese the ROI to 1/3 of the original ROI
        deltaX = int((bbox[2] - bbox[0]) * 0.33)
        deltaY = int((bbox[3] - bbox[1]) * 0.33)
        bbox[0] = bbox[0] + deltaX
        bbox[1] = bbox[1] + deltaY
        bbox[2] = bbox[2] - deltaX
        bbox[3] = bbox[3] - deltaY

        # Calculate the average depth in the ROI. TODO: median, median /w bins, mode
        cnt = 0.0
        sum = 0.0
        for x in range(bbox[2] - bbox[0]):
            for y in range(bbox[3] - bbox[1]):
                depthPixel = croppedDepth[bbox[1] + y][bbox[0] + x]
                if DEPTH_THRESH_LOW < depthPixel and depthPixel < DEPTH_THRESH_HIGH:
                    cnt+=1.0
                    sum+=depthPixel

        averageDepth = sum / cnt if 0 < cnt else 0
        # print(f"Average depth: {averageDepth}")

        # Palm detection centroid
        centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0]
        centroidY = int((bbox[3] - bbox[1]) / 2) + bbox[1]

        mid = int(croppedDepth.shape[0] / 2) # middle of the depth img
        bb_x_pos = centroidX - mid
        bb_y_pos = centroidY - mid

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth;
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        # print(f"X: {x}mm, Y: {y} mm, Z: {z} mm")
        return (x,y,z, centroidX, centroidY)

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    def calc_spatial_distance(self, spatialCoords):
        x,y,z, centroidX, centroidY = spatialCoords

        for det in self.detections:
            # Ignore detections that aren't considered dangerous
            if labelMap[det.label] not in LABEL_WARNING: continue

            self.distance = math.sqrt((det.spatialCoordinates.x-x)**2 + (det.spatialCoordinates.y-y)**2 + (det.spatialCoordinates.z-z)**2)

            height = self.frame.shape[0]
            x1 = int(det.xmin * height)
            x2 = int(det.xmax * height)
            y1 = int(det.ymin * height)
            y2 = int(det.ymax * height)
            objectCenterX = int((x1+x2)/2)
            objectCenterY = int((y1+y2)/2)
            cv2.line(self.debug_frame, (centroidX, centroidY), (objectCenterX, objectCenterY), (50,220,100), 4)

            if self.distance < WARNING_DIST:
                # Color dangerous objects in red
                sub_img = self.debug_frame[y1:y2, x1:x2]
                red_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                # Setting blue/green to 0
                red_rect[:,:,0] = 0
                red_rect[:,:,1] = 0
                res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                # Putting the image back to its position
                self.debug_frame[y1:y2, x1:x2] = res
                # Print twice to appear in bold
                cv2.putText(img=self.debug_frame,text="Danger",org=(100,int(height/3)),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.7,color=(0,0,255))
                cv2.putText(img=self.debug_frame,text="Danger",org=(101,int(height/3)),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.7,color=(0,0,255))
        cv2.imshow("color", self.debug_frame)

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

    def draw_detections(self, frame):
        color = (250,0,0)
        for detection in self.detections:
            if labelMap[detection.label] not in LABEL_WARNING: continue
            height = frame.shape[0]
            width  = frame.shape[1]
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

    def calc_angle(self, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (self.depthWidth / 2.0))

    def crop_to_rect(self, frame):
        height = frame.shape[0]
        width  = frame.shape[1]
        delta = int((width-height) / 2)
        # print(height, width, delta)
        return frame[0:height, delta:width-delta]

    def draw_palm_detection(self, palm_coords):
        if palm_coords is None: return None

        color = (10, 245, 10)
        for bbox in palm_coords:
            self.draw_bbox(bbox, color)
            spatialCoords = self.calc_spatials(bbox, self.depth)
            x,y,z,cx,cy = spatialCoords
            cv2.putText(self.debug_frame, f"X: {int(x)} mm", (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.debug_frame, f"Y: {int(y)} mm", (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.debug_frame, f"Z: {int(z)} mm", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            return spatialCoords

    def parse(self):
        self.debug_frame = self.frame.copy()

        # Parse palm detection output
        palm_coords = self.palmDetection.run_palm(
            self.debug_frame,
            self.palm_nn.tryGet())
        # Calculate and draw spatial coordinates of the palm
        spatialCoords = self.draw_palm_detection(palm_coords)
        # Calculate distance, show warning
        if spatialCoords is not None:
            self.calc_spatial_distance(spatialCoords)

        # Mobilenet detections
        self.draw_detections(self.debug_frame)
        # Put text 3 times for the bold appearance
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(50,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(51,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(52,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.imshow("color", self.debug_frame)

        if self.depthFrameColor is not None:
            self.draw_detections(self.depthFrameColor)
            cv2.imshow("depth", self.depthFrameColor)

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
stereo.setConfidenceThreshold(245)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
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

with dai.Device(pipeline) as device:
    humanMachineSafety = HumanMachineSafety()
    humanMachineSafety.create_queues(device)
    humanMachineSafety.run()

