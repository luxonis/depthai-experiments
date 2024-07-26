import depthai as dai
import asyncio
import rerun as rr
from host_rerun import Rerun

async def main():
    device = dai.Device()
    with dai.Pipeline(device) as pipeline:

        print("Creating pipeline...")
        cam = pipeline.create(dai.node.ColorCamera).build()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(1, 3)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.initialControl.setManualFocus(130)

        left = pipeline.create(dai.node.MonoCamera)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        right = pipeline.create(dai.node.MonoCamera)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)

        stereo.initialConfig.setSubpixelFractionalBits(5)
        stereo.initialConfig.postProcessing.speckleFilter.enable = False
        stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 50
        stereo.initialConfig.postProcessing.temporalFilter.enable = True
        stereo.initialConfig.postProcessing.spatialFilter.enable = True
        stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
        stereo.initialConfig.postProcessing.spatialFilter.numIterations = 1
        stereo.initialConfig.postProcessing.thresholdFilter.minRange = 400
        stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 200000
        stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 1

        rerun = pipeline.create(Rerun).build(
            color=cam.isp,
            depth=stereo.depth,
            device=device
        )
        rerun.inputs["color"].setBlocking(False)
        rerun.inputs["color"].setMaxSize(1)
        rerun.inputs["depth"].setBlocking(False)
        rerun.inputs["depth"].setMaxSize(1)

        print("Pipeline created.")
        pipeline.run()

if __name__ == "__main__":
    rr.init("Rerun")
    rr.spawn(memory_limit="2GB")
    asyncio.run(main())
