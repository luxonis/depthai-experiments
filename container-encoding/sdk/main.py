from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='4K', encode='h264')
    oak.record(color.out.main, path='records')

    oak.start(blocking=True)
