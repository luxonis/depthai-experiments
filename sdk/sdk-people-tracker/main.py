import cv2
from depthai import TrackerType, TrackerIdAssignmentPolicy, Tracklet

from depthai_sdk import OakCamera, BboxStyle, TextPosition
from depthai_sdk.callback_context import CallbackContext

THRESHOLD = 0.25

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

    if abs(deltaX) > abs(deltaY) and abs(deltaX) > THRESHOLD:
        direction = 'left' if 0 > deltaX else 'right'
        counter[direction] += 1
        print(f'Person moved {direction}')
    elif abs(deltaY) > abs(deltaX) and abs(deltaY) > THRESHOLD:
        direction = 'up' if 0 > deltaY else 'down'
        counter[direction] += 1
        print(f'Person moved {direction}')


def callback(ctx: CallbackContext):
    packet = ctx.packet
    visualizer_ = ctx.visualizer

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

            if 5 < tracked_objects[str(t.id)]['lostCnt'] and 'lost' not in tracked_objects[str(t.id)]:
                tracklet_removed(tracked_objects[str(t.id)], get_centroid(t.roi))
                tracked_objects[str(t.id)]['lost'] = True

        elif (t.status == Tracklet.TrackingStatus.REMOVED) and 'lost' not in tracked_objects[str(t.id)]:
            tracklet_removed(tracked_objects[str(t.id)], get_centroid(t.roi))

        counter_str = f'Up: {counter["up"]}\nDown: {counter["down"]}\n' \
                      f'Left: {counter["left"]}\nRight: {counter["right"]}'
        visualizer_.add_text(counter_str, position=TextPosition.BOTTOM_LEFT)
        visualizer_.draw(packet.frame)

        cv2.imshow('People tracking', packet.frame)


with OakCamera(replay='demo/example_02.mp4') as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('person-detection-retail-0013', color, nn_type='mobilenet', tracker=True)

    nn.config_tracker(tracker_type=TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
                      track_labels=[1],
                      assignment_policy=TrackerIdAssignmentPolicy.SMALLEST_ID)

    visualizer = oak.visualize(nn.out.tracker, callback=callback, fps=True)
    visualizer.detections(
        hide_label=True,
        bbox_style=BboxStyle.ROUNDED_RECTANGLE
    )

    oak.start(blocking=True)
