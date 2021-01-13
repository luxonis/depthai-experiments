import cv2
import sys
import numpy as np
import depthai as dai
from time import sleep

'''
This example attaches a NeuralNetwork node directly to the SPI output. The corresponding ESP32 example shows how to decode it.

Make sure you have something to handle the SPI protocol on the other end! See the included ESP32 example. 
'''
def create_spi_demo_pipeline(nnPath):
    print("Creating SPI pipeline: ")
    print("COLOR CAM -> DetectionNetwork -> SPI OUT")

    pipeline = dai.Pipeline()

    # set up NN node
    nn1 = pipeline.createNeuralNetwork()
    nn1.setBlobPath(nnPath)

    # set up color camera and link to NN node
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(300, 300)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setCamId(0)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.preview.link(nn1.input)

    # set up SPI out node and link to nn1
    spiOut = pipeline.createSPIOut()
    spiOut.setStreamName("spimetaout")
    spiOut.setBusId(0)
    nn1.out.link(spiOut.input)

    return pipeline


def test_pipeline():
    nnBlobPath="mobilenet-ssd.blob.sh8cmx8NCE1"
    if len(sys.argv) >= 2:
        nnBlobPath = sys.argv[1]
    pipeline = create_spi_demo_pipeline(nnBlobPath)

    print("Creating DepthAI device")
    if 1:
        device = dai.Device(pipeline)
    else: # For debug mode, with the firmware already loaded
        found, device_info = dai.XLinkConnection.getFirstDevice(
                dai.XLinkDeviceState.X_LINK_UNBOOTED)
        if found:
            device = dai.Device(pipeline, device_info)
        else:
            raise RuntimeError("Device not found")
    print("Starting pipeline")
    device.startPipeline()

    print("Pipeline is running. See connected SPI device for output!")

    while True:
        sleep(1)

    print("Closing device")
    del device

test_pipeline()
