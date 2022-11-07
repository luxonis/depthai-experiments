from depthai_sdk import OakCamera, AspectRatioResizeMode

with OakCamera() as oak:
    color = oak.create_camera('color', out='color')
    nn = oak.create_nn('mobilenet-ssd', color, out='dets')
    nn.config_nn(passthroughOut=True) # Also display passthrough frame
    nn.set_aspect_ratio_resize_mode(AspectRatioResizeMode.LETTERBOX)
    oak.create_visualizer([color, nn], fps=True)
    oak.start(blocking=True)