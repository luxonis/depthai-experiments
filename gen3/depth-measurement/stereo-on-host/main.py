import depthai as dai

from host_nodes.host_display import Display
from host_nodes.host_stereo_sgbm import StereoSGBM


device = dai.Device()
with dai.Pipeline(device) as pipeline:
    # monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    # monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
   
    # left_out = monoLeft.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.GRAY8)
    # right_out = monoRight.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.GRAY8)
    
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(cams))
    
    calibObj = device.readCalibration() 

    host = pipeline.create(StereoSGBM).build(
        monoLeftOut=mono_left.out,
        monoRightOut=mono_right.out,
        calibObj=calibObj,
        monoCamera=mono_left
    )
    
    mono_left = pipeline.create(Display).build(frame=host.mono_left)
    mono_left.setName("left")
    
    mono_right = pipeline.create(Display).build(frame=host.mono_right)
    mono_right.setName("right")
    
    disparity = pipeline.create(Display).build(frame=host.disparity_out)
    disparity.setName("disparity")
    
    rectified_left = pipeline.create(Display).build(frame=host.rectified_left)
    rectified_left.setName("rectified left")
    
    rectified_right = pipeline.create(Display).build(frame=host.rectified_right)
    rectified_right.setName("rectified right")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")