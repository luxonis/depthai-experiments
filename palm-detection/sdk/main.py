from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    nn = oak.create_nn('palm_detection_128x128', color)

    visualizer = oak.visualize(nn.out.main)
    visualizer.detections(hide_label=True, thickness=3).text(auto_scale=False)
    oak.start(blocking=True)
