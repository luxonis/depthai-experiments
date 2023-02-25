from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn.config_nn(resize_mode='letterbox')
    oak.visualize([nn, nn.out.passthrough], fps=True)
    oak.start(blocking=True)