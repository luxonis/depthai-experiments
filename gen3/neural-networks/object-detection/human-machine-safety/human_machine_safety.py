import math
import numpy as np
import cv2
import depthai as dai

from palm_detection import PalmDetection


class HumanMachineSafety(dai.node.HostNode):
    def __init__(self):
        self._palmDetection = PalmDetection()

        self._distance = 1000

        # Required information for calculating spatial coordinates on the host
        self._monoHFOV = np.deg2rad(73.5)
        self._depthWidth = 1080.0
        self._depth_thresh_low = 500
        self._depth_thresh_high = 3000
        self._warning_dist = 300

        super().__init__()


    def build(self, in_rgb: dai.Node.Output, in_det: dai.Node.Output, in_depth: dai.Node.Output, palm_in: dai.Node.Output, label_map: list[str], dangerous_objects: list[str]) -> "HumanMachineSafety":
        self.link_args(in_rgb, in_det, in_depth, palm_in)
        self.sendProcessingToPipeline(True)
        self._label_map = label_map
        self._dangerous_objects = dangerous_objects
        return self


    def set_depth_thresh_low(self, value: int) -> None:
        self._depth_thresh_low = value

    
    def set_depth_thresh_high(self, value: int) -> None:
        self._depth_thresh_high = value


    def set_warning_dist(self, value: int) -> None:
        self._warning_dist = value


    def process(self, in_rgb: dai.ImgFrame, in_det: dai.ImgDetections, in_depth: dai.ImgFrame, palm_in: dai.NNData) -> None:
        detections = []
        depth_frame = None
        depth_frame_color = None
        frame = None

        if in_rgb is not None:
            frame = self._crop_to_rect(in_rgb.getCvFrame())

        # Check for mobilenet detections
        if in_det is not None:
            detections = in_det.detections

        if in_depth is not None:
            depth_frame = self._crop_to_rect(in_depth.getFrame())
            depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_JET)

        if palm_in is not None and frame is not None and depth_frame is not None:
            try:
                self.parse(
                    in_palm_detections=palm_in,
                    detections=detections,
                    frame=frame,
                    depth=depth_frame,
                    depthColored=depth_frame_color)
            except StopIteration:
                self.stopPipeline()


    def _crop_to_rect(self, frame: np.ndarray) -> np.ndarray:
        height = frame.shape[0]
        width  = frame.shape[1]
        delta = int((width-height) / 2)
        return frame[0:height, delta:width-delta]


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

        if xmin == xmax or ymin == ymax: # Box of size zero
            return None

        # Calculate the average depth in the ROI.
        depthROI = depth[ymin:ymax, xmin:xmax]
        inThreshRange = (self._depth_thresh_low < depthROI) & (depthROI < self._depth_thresh_high)

        averageDepth = averaging_method(depthROI[inThreshRange])

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

        return (x,y,z, centroidX, centroidY)


    def calc_spatial_distance(self, spatialCoords, frame: np.ndarray, detections: list[dai.SpatialImgDetection]):
        x,y,z, centroidX, centroidY = spatialCoords
        annotate = self._annotate_fun(frame, (0, 0, 25), fontScale=1.7)

        for det in detections:
            # Ignore detections that aren't considered dangerous
            if self._label_map[det.label] not in self._dangerous_objects: continue

            self._distance = math.sqrt((det.spatialCoordinates.x-x)**2 + (det.spatialCoordinates.y-y)**2 + (det.spatialCoordinates.z-z)**2)

            height = frame.shape[0]
            x1 = int(det.xmin * height)
            x2 = int(det.xmax * height)
            y1 = int(det.ymin * height)
            y2 = int(det.ymax * height)
            objectCenterX = int((x1+x2)/2)
            objectCenterY = int((y1+y2)/2)
            cv2.line(frame, (centroidX, centroidY), (objectCenterX, objectCenterY), (50,220,100), 4)

            if self._distance < self._warning_dist:
                # Color dangerous objects in red
                sub_img = frame[y1:y2, x1:x2]
                red_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                # Setting blue/green to 0
                red_rect[:,:,0] = 0
                red_rect[:,:,1] = 0
                res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                # Putting the image back to its position
                frame[y1:y2, x1:x2] = res
                # Print twice to appear in bold
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


    def draw_detections(self, frame: np.ndarray, detections: list[dai.SpatialImgDetection]):
        color = (250,0,0)
        annotate = self._annotate_fun(frame, (0, 0, 25))

        for detection in detections:
            if self._label_map[detection.label] not in self._dangerous_objects: continue
            height = frame.shape[0]
            width  = frame.shape[1]
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            offsetX = x1 + 10
            annotate("{:.2f}".format(detection.confidence*100), (offsetX, y1 + 35))
            annotate(f"X: {int(detection.spatialCoordinates.x)} mm", (offsetX, y1 + 50))
            annotate(f"Y: {int(detection.spatialCoordinates.y)} mm", (offsetX, y1 + 65))
            annotate(f"Z: {int(detection.spatialCoordinates.z)} mm", (offsetX, y1 + 80))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            try:
                label = self._label_map[detection.label]
            except:
                label = detection.label

            annotate(str(label), (offsetX, y1 + 20))


    def calc_angle(self, offset):
        return math.atan(math.tan(self._monoHFOV / 2.0) * offset / (self._depthWidth / 2.0))


    def draw_palm_detection(self, palm_coords, depth):
        if palm_coords is None:
            return None

        color = (10,245,10)
        annotate = self._annotate_fun(self.debug_frame, color)

        for bbox in palm_coords:
            spatialCoords = self.calc_spatials(bbox, depth)
            if spatialCoords is None: # Box of size 0
                continue
            self.draw_bbox(bbox, color)
            x,y,z,cx,cy = spatialCoords
            annotate(f"X: {int(x)} mm", (bbox[0], bbox[1]))
            annotate(f"Y: {int(y)} mm", (bbox[0], bbox[1] + 15))
            annotate(f"Z: {int(z)} mm", (bbox[0], bbox[1] + 30))
            return spatialCoords


    def parse(self, in_palm_detections, detections, frame, depth, depthColored):
        self.debug_frame = frame.copy()
        self.depthFrameColor = depthColored
        annotate = self._annotate_fun(self.debug_frame, (50,220,110), fontScale=1.4)

        # Parse palm detection output
        palm_coords = self._palmDetection.run_palm(
            self.debug_frame,
            in_palm_detections)
        # Calculate and draw spatial coordinates of the palm
        spatialCoords = self.draw_palm_detection(palm_coords, depth)
        # Calculate distance, show warning
        if spatialCoords is not None:
            self.calc_spatial_distance(spatialCoords,self.debug_frame, detections)

        # Mobilenet detections
        self.draw_detections(self.debug_frame, detections)
        # Put text 3 times for the bold appearance
        annotate(f"Distance: {int(self._distance)} mm", (50,700))
        annotate(f"Distance: {int(self._distance)} mm", (51,700))
        annotate(f"Distance: {int(self._distance)} mm", (52,700))
        cv2.imshow("color", self.debug_frame)

        if self.depthFrameColor is not None:
            self.draw_detections(self.depthFrameColor, detections)
            cv2.imshow("depth", self.depthFrameColor)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration()
        
    
    def _annotate_fun(self, img, color, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, **kwargs):
        def fun(text, pos):
            cv2.putText(img, text, pos, fontFace, fontScale, color, **kwargs)
        return fun