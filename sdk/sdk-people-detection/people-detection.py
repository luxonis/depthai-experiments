from depthai_sdk import OakCamera, BboxStyle

with OakCamera(replay='people-tracking-above-04') as oak:
    camera = oak.create_camera('color')

    det = oak.create_nn('person-detection-retail-0013', camera, tracker=True)
    det.config_nn(conf_threshold=0.1)

    visualizer = oak.visualize(det.out.tracker)
    visualizer.detections(bbox_style=BboxStyle.CORNERS, color=(232, 36, 87), hide_label=True)
    visualizer.tracking(line_color=(114, 210, 252), deletion_lost_threshold=2, line_thickness=3, fading_tails=3)

    oak.start(blocking=True)
