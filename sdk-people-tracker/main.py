import argparse

import blobconverter
import cv2
from depthai import TrackerType, TrackerIdAssignmentPolicy, Tracklet

from depthai_sdk import OakCamera, BboxStyle, TrackerPacket, Visualizer, TextPosition

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help='.blob path')
parser.add_argument('-vid', '--video', type=str,
                    help='Path to video file to be used for inference (conflicts with -cam)')
parser.add_argument('-spi', '--spi', action='store_true', default=False, help='Send tracklets to the MCU via SPI')
parser.add_argument('-cam', '--camera', action='store_true',
                    help='Use DepthAI RGB camera for inference (conflicts with -vid)')
parser.add_argument('-t', '--threshold', default=0.25, type=float,
                    help='Minimum distance the person has to move (across the x/y axis) to be considered a real movement')
args = parser.parse_args()

nn_path = args.nn or blobconverter.from_zoo(name='person-detection-retail-0013', shaves=7)

tracked_objects = {}
counter = {'up': 0, 'down': 0, 'left': 0, 'right': 0}


def get_centroid(roi):
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    return ((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)


def tracklet_removed(tracklet, coords2):
    coords1 = tracklet['coords']
    deltaX = coords2[0] - coords1[0]
    deltaY = coords2[1] - coords1[1]

    if abs(deltaX) > abs(deltaY) and abs(deltaX) > args.threshold:
        direction = 'left' if 0 > deltaX else 'right'
        counter[direction] += 1
        print(f'Person moved {direction}')
    elif abs(deltaY) > abs(deltaX) and abs(deltaY) > args.threshold:
        direction = 'up' if 0 > deltaY else 'down'
        counter[direction] += 1
        print(f'Person moved {direction}')


def callback(packet: TrackerPacket, visualizer_: Visualizer, **kwargs):
    for t in packet.daiTracklets.tracklets:
        # If new tracklet, save its centroid
        if t.status == Tracklet.TrackingStatus.NEW:
            tracked_objects[str(t.id)] = {}  # Reset
            tracked_objects[str(t.id)]['coords'] = get_centroid(t.roi)

        elif t.status == Tracklet.TrackingStatus.TRACKED:
            tracked_objects[str(t.id)]['lostCnt'] = 0

        elif t.status == Tracklet.TrackingStatus.LOST:
            tracked_objects[str(t.id)]['lostCnt'] += 1
            # If tracklet has been 'LOST' for more than 10 frames, remove it

            if 10 < tracked_objects[str(t.id)]['lostCnt'] and 'lost' not in tracked_objects[str(t.id)]:
                tracklet_removed(tracked_objects[str(t.id)], get_centroid(t.roi))
                tracked_objects[str(t.id)]['lost'] = True

        elif (t.status == Tracklet.TrackingStatus.REMOVED) and 'lost' not in tracked_objects[str(t.id)]:
            tracklet_removed(tracked_objects[str(t.id)], get_centroid(t.roi))

        counter_str = f'Up: {counter["up"]}\nDown: {counter["down"]}\n' \
                      f'Left: {counter["left"]}\nRight: {counter["right"]}'
        visualizer_.add_text(counter_str, position=TextPosition.BOTTOM_LEFT)
        visualizer_.draw(packet.frame)

        cv2.imshow('People tracking', packet.frame)


with OakCamera(replay='demo/example_01.mp4') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn(nn_path, color, type='mobilenet', tracker=True)

    nn.config_tracker(type=TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
                      trackLabels=[1],
                      assignmentPolicy=TrackerIdAssignmentPolicy.SMALLEST_ID)

    visualizer = oak.visualize(nn.out.tracker, callback=callback, fps=True)
    visualizer.detections(
        hide_label=True,
        bbox_style=BboxStyle.ROUNDED_RECTANGLE
    )

    oak.start(blocking=True)
