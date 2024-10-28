import depthai as dai

from host_node.host_display import Display
from host_stereo_sgbm import StereoSGBM

RESOLUTION_SIZE = (640, 480)

device = dai.Device()
with dai.Pipeline(device) as pipeline:
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    
    left = mono_left.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.NV12)
    right = mono_right.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.NV12)
    
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(cams))
    
    calibObj = device.readCalibration()

    host = pipeline.create(StereoSGBM).build(
        monoLeftOut=left,
        monoRightOut=right,
        calibObj=calibObj,
        resolution=RESOLUTION_SIZE
    )
    host.inputs["monoLeft"].setBlocking(False)
    host.inputs["monoLeft"].setMaxSize(2)
    host.inputs["monoRight"].setBlocking(False)
    host.inputs["monoRight"].setMaxSize(2)
    
    mono_left = pipeline.create(Display).build(frames=host.mono_left)
    mono_left.setName("left")
    
    mono_right = pipeline.create(Display).build(frames=host.mono_right)
    mono_right.setName("right")
    
    disparity = pipeline.create(Display).build(frames=host.disparity_out)
    disparity.setName("disparity")
    
    rectified_left = pipeline.create(Display).build(frames=host.rectified_left)
    rectified_left.setName("rectified left")
    
    rectified_right = pipeline.create(Display).build(frames=host.rectified_right)
    rectified_right.setName("rectified right")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")