from depthai_sdk import OakCamera, AspectRatioResizeMode

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.LETTERBOX)
    oak.visualize([nn, nn.out.passthrough], fps=True)
    oak.start(blocking=True)