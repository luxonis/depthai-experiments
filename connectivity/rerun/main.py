import depthai as dai
import asyncio
import rerun as rr

from host_rerun import Rerun


async def main():
    device = dai.Device()
    with dai.Pipeline(device) as pipeline:

        print("Creating pipeline...")
        device.setIrLaserDotProjectorIntensity(1)

        rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left.requestOutput(size=(640, 400)),
            right=right.requestOutput(size=(640, 400))
        )
        stereo.setOutputSize(1920, 1080)
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

        pcl = pipeline.create(dai.node.PointCloud)
        stereo.depth.link(pcl.inputDepth)

        pipeline.create(Rerun).build(
            color=rgb.requestOutput((1920, 1080)),
            pcl=pcl.outputPointCloud
        )

        print("Pipeline created.")
        pipeline.run()

if __name__ == "__main__":
    rr.init("Rerun")
    rr.spawn(memory_limit="2GB")
    asyncio.run(main())
