import numpy as np
import scipy.special
import cv2

def frameNormHeight(frame, bbox, pad):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[0]
    return (np.clip(np.array(bbox), 0, 1) * normVals + np.array([pad, 0, pad, 0])).astype(int)

def drawDets(frame, detections, labels, pad = 176):
    color = (255, 0, 0)
    for detection in detections:
        # normalize by height because center cropped, then pad x so detection are position correctly on the input image
        bbox = frameNormHeight(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax), pad)
        cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x / 100)} m", (bbox[0] + 10, bbox[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y / 100)} m", (bbox[0] + 10, bbox[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        #cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z / 100)} m", (bbox[0] + 10, bbox[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


def pred_to_lines(predictions, row_anchors, griding_num = 100, input_size = (288, 512), target_size = (1080, 1920)):

    # get columns
    col_sample = np.linspace(0, input_size[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    preds = predictions[:, ::-1, :]

    # softmax by rows
    preds = scipy.special.softmax(preds, axis=0)
    
    # normalize without background prediction
    prob = preds[:-1, :, :] / np.sum(preds[:-1, :, :], axis=0)

    # get indices for expected value
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)

    # compute expected value for location
    loc = np.sum(prob * idx, axis=0)

    # filter out where background > 0.5
    loc[preds[-1, :, :] > 0.5] = 0
    
    # lanes
    lanes = np.ones((preds.shape[-1], preds.shape[1], 2)) * -1 # (lanes, rows, coordinates)

    cls_num_per_lane = len(row_anchors)

    for i in range(loc.shape[1]):
        # if more than two points for this lane
        if np.sum(loc[:, i] != 0) > 2:
            # go over rows
            for k in range(loc.shape[0]):
                if loc[k, i] > 0:
                    x = (loc[k, i] - 1) * col_sample_w * target_size[1] / input_size[1] # loc * width * normalize
                    y = row_anchors[cls_num_per_lane-1-k] * target_size[0] / input_size[0] # loc * normalize
                    lanes[i, k] = np.array([x,y])
    #print(lanes)
    return lanes


def draw_points(image, lanes, colors):
    for i in range(lanes.shape[0]):
        for r in range(lanes.shape[1]):
            x, y = int(lanes[i, r, 0]), int(lanes[i, r, 1])
            if x > -1:
                color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
                cv2.circle(image, (x,y) ,3,color,-1)