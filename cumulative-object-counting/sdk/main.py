import cv2
import numpy as np
from depthai import TrackerType, TrackerIdAssignmentPolicy, Tracklet

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import TextPosition, BboxStyle

tracked_objects = {}
counter = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

ROI_POS = 0.5
AXIS = 1


class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False


def get_centroid(roi):
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    return ((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)


def callback(packet, visualizer):
    height, width = packet.frame.shape[:2]

    for t in packet.daiTracklets.tracklets:
        to = tracked_objects.get(t.id, None)

        # calculate centroid
        roi = t.roi.denormalize(width, height)
        x1 = int(roi.topLeft().x)
        y1 = int(roi.topLeft().y)
        x2 = int(roi.bottomRight().x)
        y2 = int(roi.bottomRight().y)
        centroid = (int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1))

        # If new tracklet, save its centroid
        if t.status == Tracklet.TrackingStatus.NEW:
            to = TrackableObject(t.id, centroid)
        elif to is not None and not to.counted:
            if AXIS == 0:
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)

                if centroid[0] > ROI_POS * width and direction > 0 and np.mean(x) < ROI_POS * width:
                    counter['right'] += 1
                    to.counted = True
                elif centroid[0] < ROI_POS * width and direction < 0 and np.mean(x) > ROI_POS * width:
                    counter['left'] += 1
                    to.counted = True

            elif AXIS == 1:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)

                if centroid[1] > ROI_POS * height and direction > 0 and np.mean(y) < ROI_POS * height:
                    counter['down'] += 1
                    to.counted = True
                elif centroid[1] < ROI_POS * height and direction < 0 and np.mean(y) > ROI_POS * height:
                    counter['up'] += 1
                    to.counted = True

            to.centroids.append(centroid)

        tracked_objects[t.id] = to

        if t.status != Tracklet.TrackingStatus.LOST and t.status != Tracklet.TrackingStatus.REMOVED:
            text = 'ID {}'.format(t.id)

            cv2.putText(packet.frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(packet.frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        counter_str = f'Up: {counter["up"]}\nDown: {counter["down"]}\n' \
            if AXIS == 1 \
            else f'Left: {counter["left"]}\nRight: {counter["right"]}'

        visualizer.add_line(pt1=(0, int(ROI_POS * height)),
                            pt2=(width, int(ROI_POS * height)),
                            color=(255, 255, 255),
                            thickness=2)

        visualizer.add_text(counter_str, position=TextPosition.BOTTOM_LEFT)
        frame = visualizer.draw(packet.frame)

        cv2.imshow('People tracking', frame)


with OakCamera(replay='../demo/example_01.mp4') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color, nn_type='mobilenet', tracker=True)

    nn.config_tracker(tracker_type=TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
                      assignment_policy=TrackerIdAssignmentPolicy.SMALLEST_ID)

    visualizer = oak.visualize(nn.out.tracker, callback=callback, fps=True)
    visualizer.detections(
        hide_label=True,
        bbox_style=BboxStyle.ROUNDED_RECTANGLE
    )

    oak.start(blocking=True)
