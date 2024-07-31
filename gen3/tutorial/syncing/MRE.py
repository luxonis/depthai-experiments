#!/usr/bin/env python3

import cv2
import depthai as dai

class Display(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()


    def build(self, cam_isp : dai.Node.Output, imu_out : dai.Node.Output) -> "Display":

        self.link_args(cam_isp, imu_out) # doesnt show output, paired with the first process method 
        # self.link_args(cam_isp) # shows output, works fine, paired with the second process method
        self.sendProcessingToPipeline(True)
        return self
    
    # 2 process methods to showcase that the problem is with IMU node

    def process(self, in_frame : dai.Buffer, imu : dai.Buffer) -> None:
        assert(isinstance(in_frame, dai.ImgFrame))

        cv2.imshow("frame", in_frame.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
    

    # def process(self, in_frame : dai.Buffer) -> None:
    #     assert(isinstance(in_frame, dai.ImgFrame))

    #     cv2.imshow("frame", in_frame.getCvFrame())

    #     if cv2.waitKey(1) == ord('q'):
    #         self.stopPipeline()


with dai.Pipeline() as pipeline:

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)
    camRgb.setIspScale(2, 3)

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 360)
    imu.setBatchReportThreshold(10)
    imu.setMaxBatchReports(10)

    ### should be synced but program doesnt get to process method ###
    sync_node = pipeline.create(dai.node.Sync)
    camRgb.isp.link(sync_node.inputs['rgb'])
    imu.out.link(sync_node.inputs['imu'])

    demux = pipeline.create(dai.node.MessageDemux)
    
    sync_node.out.link(demux.input)

    pipeline.create(Display).build(
        cam_isp=demux.outputs['rgb'],
        imu_out=demux.outputs['imu']
    )
    ### should be synced but program doesnt get to process method ###

    ### not synced, imu timestamp is always 0 .. weird ###
    # pipeline.create(Display).build(
    #     cam_isp=camRgb.isp,
    #     imu_out=imu.out
    # )
    ### not synced, imu timestamp is always 0 .. weird ###

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
