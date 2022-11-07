from depthai_sdk import OakCamera


def callback(packet):
    print(packet)


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('models/model.blob', color)

    oak.callback(nn, callback=callback)

    oak.start(blocking=True)
