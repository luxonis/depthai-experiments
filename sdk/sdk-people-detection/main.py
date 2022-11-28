from depthai_sdk import OakCamera, BboxStyle

with OakCamera(replay='people.mp4') as oak:
    camera = oak.create_camera('color', fps=30, encode='h264')

    det = oak.create_nn('person-detection-retail-0013', camera, nn_type='mobilenet')
    det.config_nn(conf_threshold=0.1)

    visualizer = oak.visualize(det.out.encoded)
    visualizer.detections(bbox_style=BboxStyle.CORNERS, color=(232, 36, 87), hide_label=True)

    oak.start(blocking=True)
