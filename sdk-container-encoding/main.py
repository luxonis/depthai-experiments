import sys

from depthai_sdk import OakCamera

with OakCamera() as oak:
    codec = sys.argv[1] if sys.argv else None or 'h265'

    color = oak.create_camera('color', resolution='4K', encode='h264')
    oak.record(color.out.main, path='records')

    oak.start(blocking=True)
