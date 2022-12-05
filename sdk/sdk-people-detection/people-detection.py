from depthai_sdk import OakCamera, BboxStyle

with OakCamera(replay='people-tracking-above-04') as oak:
    camera = oak.create_camera('color', fps=30, encode='h264')

    det = oak.create_nn('person-detection-retail-0013', camera)
    det.config_nn(confThreshold=0.1)

    visualizer = oak.visualize(det.out.main)
    visualizer.detections(bbox_style=BboxStyle.CORNERS, color=(232, 36, 87), hide_label=True)

    oak.start(blocking=True)
