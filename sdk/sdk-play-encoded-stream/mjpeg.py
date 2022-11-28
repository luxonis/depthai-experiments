from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', encode='mjpeg', fps=60)
    oak.visualize(color.out.encoded)
    oak.start(blocking=True)
