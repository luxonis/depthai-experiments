import depthai as dai
import numpy as np
import cv2

CONF_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.3

class East(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.passthrough = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)])

    def build(self, video: dai.Node.Output, nn: dai.Node.Output) -> "East":
        self.link_args(video, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, video: dai.ImgFrame, detections: dai.NNData) -> None:
        frame = video.getCvFrame()

        layer_names = detections.getAllLayerNames()
        scores = detections.getTensor(layer_names[0]).flatten().reshape(1, 1, 64, 64)
        geometry1 = detections.getTensor(layer_names[1]).flatten().reshape(1, 4, 64, 64)
        geometry2 = detections.getTensor(layer_names[2]).flatten().reshape(1, 1, 64, 64)

        bboxes, confs, angles = decode_predictions(scores, geometry1, geometry2)
        boxes, angles = non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))

        output_dets = dai.ImgDetections()
        output_dets.setTimestamp(detections.getTimestamp())
        output_dets.setTimestampDevice(detections.getTimestampDevice())
        # Workaround to append to output_dets.detections because it returns a copy, not a reference
        dets = []

        rotated_rectangles = [get_cv_rotated_rect(bbox, -angle) for (bbox, angle) in zip(boxes, angles)]
        for rect in rotated_rectangles:
            # Detections are done on 256x256 frames, video is 1024x1024
            rect[0][0] = rect[0][0] * 4
            rect[0][1] = rect[0][1] * 4
            rect[1][0] = rect[1][0] * 4
            rect[1][1] = rect[1][1] * 4

            # Workaround to use dai.ImgDetection to act like rotated rectangle
            (x1, y1), (x2, y2), conf = rect

            detection = dai.ImgDetection()
            detection.xmin = x1
            detection.ymin = y1
            detection.xmax = x2
            detection.ymax = y2
            detection.confidence = conf
            dets.append(detection)

            # Draw detection crop area on input frame
            points = np.int64(cv2.boxPoints(rect))
            cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_8)

        output_dets.detections = dets
        self.output.send(output_dets)

        # A copy needs to be created because message queues are stored as pointers in the C++ depthai implementation
        #  and changing the type to BGR888i would change the message going to the neural network
        output_frame = dai.ImgFrame()
        output_frame.setType(dai.ImgFrame.Type.BGR888i)
        output_frame.setFrame(frame)
        output_frame.setWidth(video.getWidth())
        output_frame.setHeight(video.getHeight())
        output_frame.setTimestamp(video.getTimestamp())
        output_frame.setTimestampDevice(video.getTimestampDevice())
        self.passthrough.send(output_frame)


def get_cv_rotated_rect(bbox, angle):
    x0, y0, x1, y1 = bbox
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x = x0 + width * 0.5
    y = y0 + height * 0.5
    return [x.tolist(), y.tolist()], [width.tolist(), height.tolist()], np.rad2deg(angle)

def non_max_suppression(boxes, probs=None, angles=None):

    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    filtered = []

    # If the bounding boxes are integers, convert them to floats since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and grab the indexes to sort
    #  (in the case that no probabilities are provided, simply sort on the bottom-left y-coordinate)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2 if probs is None else probs)[::-1]

    while len(idxs) > 0:
        # Grab the first index in the indexes list and add the index value to the list of picked indexes
        best = idxs[0]
        filtered.append(best)

        # Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to best
        overlap_xmins = np.maximum(x1[best], x1[idxs[1:]])
        overlap_ymins = np.maximum(y1[best], y1[idxs[1:]])
        overlap_xmaxs = np.minimum(x2[best], x2[idxs[1:]])
        overlap_ymaxs = np.minimum(y2[best], y2[idxs[1:]])

        overlap_widths = np.maximum(1, (overlap_xmaxs - overlap_xmins))
        overlap_heights = np.maximum(1, (overlap_ymaxs - overlap_ymins))
        overlap_areas = overlap_widths * overlap_heights

        # Calculating all intersection over unions
        IOUs = overlap_areas / (areas[best] + areas[idxs[1:]] - overlap_areas)

        # Deletes them if the ratio is greater than threshold
        delete_idxs = np.concatenate(([0], np.where(IOUs > OVERLAP_THRESHOLD)[0] + 1))
        idxs = np.delete(idxs, delete_idxs)

    # Return only the bounding boxes that were picked
    return boxes[filtered].astype("int"), angles[filtered]

def decode_predictions(scores, geometry1, geometry2):
    # Grab the number of rows and columns from the scores volume, then
    #  initialize our set of bounding box rectangles and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    angles = []

    for y in range(0, numRows):
        # Extract the scores (probabilities), followed by the
        #  geometrical data used to derive potential bounding box coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry1[0, 0, y]
        xData1 = geometry1[0, 1, y]
        xData2 = geometry1[0, 2, y]
        xData3 = geometry1[0, 3, y]
        anglesData = geometry2[0, 0, y]

        for x in range(0, numCols):
            if scoresData[x] < CONF_THRESHOLD:
                continue

            # Compute the offset factor as our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            angles.append(angle)

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences, angles)
