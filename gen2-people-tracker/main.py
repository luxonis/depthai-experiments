#!/usr/bin/env python3

from depthai_sdk import OakCamera, TrackerPacket, Visualizer, TextPosition
import depthai as dai
from people_tracker import PeopleTracker
import cv2

pt = PeopleTracker()

with OakCamera(replay='people-tracking-above-03') as oak:
    color_cam = oak.create_camera('color')
    tracker = oak.create_nn('person-detection-retail-0013', color_cam, tracker=True)
    tracker.config_nn(conf_threshold=0.6)
    tracker.config_tracker(tracker_type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM, track_labels=[1])

    def cb(packet: TrackerPacket, vis: Visualizer):
        left, right, up, down = pt.calculate_tracklet_movement(packet.daiTracklets)

        vis.add_text(f"Up: {up}, Down: {down}", position=TextPosition.TOP_LEFT, size=1)
        vis.draw(packet.frame)

        cv2.imshow('People Tracker', packet.frame)

    oak.visualize(tracker.out.tracker, callback=cb)
    oak.start(blocking=True)
