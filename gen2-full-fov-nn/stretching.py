from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('yolov7tiny_coco_416x416', color)
    nn.config_nn(resize_mode='stretch')
    oak.visualize([nn, nn.out.passthrough], fps=True)
    oak.start(blocking=True)