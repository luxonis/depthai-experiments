# coding=utf-8
from pathlib import Path
import math
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS

DEPTH_THRESH_HIGH = 2000
DEPTH_THRESH_LOW = 600
WARNING_DIST = 300
# MobilenetSSD label texts
LABEL_WARNING = [5] # 5 = bottle
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }

def frame_norm(frame, *xy_vals):
    """
    nn data, being the bounding box locations, are in <0..1> range -
    they need to be normalized with frame width/height

    :param frame:
    :param xy_vals: the bounding box locations
    :return:
    """
    return (
        np.clip(np.array(xy_vals), 0, 1)
        * np.array(frame.shape[:2] * (len(xy_vals) // 2))[::-1]
    ).astype(int)

class DepthAI:
    def __init__(self,file=None):
        print("Loading pipeline...")
        self.file = file
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 1
        self.lineType = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = dai.Pipeline()

        cam = self.pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(2, 3) # To match 720P mono cameras
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.initialControl.setManualFocus(130)
        # For MobileNet NN
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setPreviewSize(300, 300)
        cam.setInterleaved(False)

        isp_xout = self.pipeline.createXLinkOut()
        isp_xout.setStreamName("cam")
        cam.isp.link(isp_xout.input)

        # For Palm-detection NN
        self.manip = self.pipeline.createImageManip()
        self.manip.initialConfig.setResize(128, 128)
        cam.preview.link(self.manip.inputImage)

        left = self.pipeline.createMonoCamera()
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)

        right = self.pipeline.createMonoCamera()
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(245)
        stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        sdn = self.pipeline.createMobileNetSpatialDetectionNetwork()
        sdn.setBlobPath("models/mobilenet-ssd_openvino_2021.2_6shave.blob")
        sdn.setConfidenceThreshold(0.5)
        sdn.input.setBlocking(False)
        sdn.setBoundingBoxScaleFactor(0.2)
        sdn.setDepthLowerThreshold(DEPTH_THRESH_LOW)
        sdn.setDepthUpperThreshold(DEPTH_THRESH_HIGH)

        cam.preview.link(sdn.input)
        stereo.depth.link(sdn.inputDepth)

        sdn_out = self.pipeline.createXLinkOut()
        sdn_out.setStreamName("det")
        sdn.out.link(sdn_out.input)

        depth_out = self.pipeline.createXLinkOut()
        depth_out.setStreamName("depth")
        sdn.passthroughDepth.link(depth_out.input)

        self.create_nns()

        print("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """
        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first:
            print("linked manip.out to model_nn.input")
            self.manip.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def crop_to_rect(self, frame):
        height = frame.shape[0]
        width  = frame.shape[1]
        delta = int((width-height) / 2)
        # print(height, width, delta)
        return frame[0:height, delta:width-delta]

    def start_pipeline(self):
        self.device = dai.Device(self.pipeline)
        print("Starting pipeline...")
        self.device.startPipeline()

        self.start_nns()

        self.vidQ = self.device.getOutputQueue(name="cam", maxSize=4, blocking=False)

        self.detQ = self.device.getOutputQueue(name="det", maxSize=4, blocking=False)
        self.depthQ = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def start_nns(self):
        pass

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

    def calc_spatial_distance(self, x,y,z, centroidX, centroidY):
        for det in self.detections:
            if det.label not in LABEL_WARNING: continue
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
                # Color chair BB in red
                sub_img = self.debug_frame[y1:y2, x1:x2]
                red_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                # Setting blue/green to 0
                red_rect[:,:,0] = 0
                red_rect[:,:,1] = 0
                res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                # Putting the image back to its position
                self.debug_frame[y1:y2, x1:x2] = res

                cv2.putText(img=self.debug_frame,text="Drink water instead",org=(100,int(height/3)),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.7,color=(0,0,255))
                cv2.putText(img=self.debug_frame,text="Drink water instead",org=(101,int(height/3)),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.7,color=(0,0,255))
        cv2.imshow("color", self.debug_frame)

    def draw_detections(self, frame):
        color = (250,0,0)
        for detection in self.detections:
            if detection.label not in LABEL_WARNING: continue
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

    def parse(self):
        self.debug_frame = self.frame.copy()

        self.parse_fun()

        self.draw_detections(self.debug_frame)
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(50,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(51,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.putText(img=self.debug_frame,text=f"Distance: {int(self.distance)} mm",org=(52,700),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.4,color=(50,220, 110))
        cv2.imshow("color", self.debug_frame)

        if self.depthFrameColor is not None:
            self.draw_detections(self.depthFrameColor)
            cv2.imshow("depth", self.depthFrameColor)

        self.fps_cam.update()
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            self.fps_cam.stop()
            self.fps_nn.stop()
            print(f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}")
            raise StopIteration()

    def parse_fun(self):
        pass

    def run_camera(self):
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
                # print(type(self.depth)) #np.ndarray
                depthFrameColor = cv2.normalize(self.depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                self.depthFrameColor = self.crop_to_rect(depthFrameColor)
                # self.drawSlc(self.depthFrameColor, self.spatials)

            # in_det = self.detQ.tryGet()

    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        self.run_camera()
        del self.device


def sigmoid(x):
    return (1.0 + np.tanh(0.5 * x)) * 0.5

def decode_boxes(raw_boxes, anchors, shape, num_keypoints):
    """
    Converts the predictions into actual coordinates using the anchor boxes.
    Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)
    x_scale, y_scale = shape

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / x_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / y_scale * anchors[:, 3]

    boxes[..., 1] = y_center - h / 2.0  # xmin
    boxes[..., 0] = x_center - w / 2.0  # ymin
    boxes[..., 3] = y_center + h / 2.0  # xmax
    boxes[..., 2] = x_center + w / 2.0  # ymax

    for k in range(num_keypoints):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = (
            raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes

def raw_to_detections(raw_box_tensor, raw_score_tensor, anchors_, shape, num_keypoints):
    """

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = decode_boxes(raw_box_tensor, anchors_, shape, num_keypoints)
    detection_scores = sigmoid(raw_score_tensor).squeeze(-1)
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i]
        scores = np.expand_dims(detection_scores[i], -1)
        output_detections.append(np.concatenate((boxes, scores), -1))
    return output_detections

def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    if angles is not None:
        return boxes[pick].astype("int"), angles[pick]
    return boxes[pick].astype("int")


class Main(DepthAI):
    def __init__(self, file=None):
        super().__init__(file)

    def create_nns(self):
        self.create_nn("models/palm_detection_openvino_2021.3_6shave.blob", "palm", first=True)

    def start_nns(self):
        self.palm_nn = self.device.getOutputQueue("palm_nn", maxSize=4, blocking=False)

    def run_palm(self):
        """
        Each palm detection is a tensor consisting of 19 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 7 key_points
            - confidence score
        :return:
        """
        shape = (128, 128)
        num_keypoints = 7
        min_score_thresh = 0.7
        anchors = np.load("anchors_palm.npy")

        nn_data = self.palm_nn.tryGet()

        if nn_data is None:
            return
        # Run the neural network
        results = to_tensor_result(nn_data)

        raw_box_tensor = results.get("regressors").reshape(-1, 896, 18)  # regress
        raw_score_tensor = results.get("classificators").reshape(-1, 896, 1)  # classification

        detections = raw_to_detections(raw_box_tensor, raw_score_tensor, anchors, shape, num_keypoints)

        self.palm_coords = [
            frame_norm(self.frame, *obj[:4])
            for det in detections
            for obj in det
            if obj[-1] > min_score_thresh
        ]

        self.palm_confs = [
            obj[-1] for det in detections for obj in det if obj[-1] > min_score_thresh
        ]

        if len(self.palm_coords) == 0:
            return

        self.palm_coords = non_max_suppression(
            boxes=np.concatenate(self.palm_coords).reshape(-1, 4),
            probs=self.palm_confs,
            overlapThresh=0.1,
        )

        color = (10, 245, 10)
        for bbox in self.palm_coords:
            self.draw_bbox(bbox, color)
            x,y,z = self.calc_spatials(bbox)
            cv2.putText(self.debug_frame, f"X: {int(x)} mm", (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.debug_frame, f"Y: {int(y)} mm", (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.debug_frame, f"Z: {int(z)} mm", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

    def calc_spatials(self, bbox):
        croppedDepth = self.crop_to_rect(self.depth)
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

        def calc_angle(offset):
            hfov = np.deg2rad(73.5) # Mono HFOV
            depthWidth = 1080.0
            ret = math.atan(math.tan(hfov / 2.0) * offset / (depthWidth / 2.0))
            return ret

        angle_x = calc_angle(bb_x_pos)
        angle_y = calc_angle(bb_y_pos)

        z = averageDepth;
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        # print(f"X: {x}mm, Y: {y} mm, Z: {z} mm")
        self.calc_spatial_distance(x,y,z, centroidX, centroidY)
        return (x,y,z)


    # @timer
    def parse_fun(self):
        self.run_palm()


if __name__ == "__main__":
    Main().run()
