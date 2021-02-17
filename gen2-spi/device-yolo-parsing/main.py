import cv2
import sys
import numpy as np
import depthai as dai
from time import sleep

'''
Basic demo of gen2 pipeline builder functionality where output of jpeg encoded images are sent out SPI rather than the typical XLink out interface.

Make sure you have something to handle the SPI protocol on the other end! See the included ESP32 example. 
'''
def create_spi_demo_pipeline(nnPath):
    print("Creating SPI pipeline: ")
    print("COLOR CAM -> DetectionNetwork -> SPI OUT")

    pipeline = dai.Pipeline()

    # testing YOLO DetectionNetwork 
    detectionNetwork = pipeline.createYoloDetectionNetwork()
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    anchors = np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    detectionNetwork.setAnchors(anchors)
    anchorMasks26 = np.array([1,2,3])
    anchorMasks13 = np.array([3,4,5])
    anchorMasks = {
        "side26": anchorMasks26,
        "side13": anchorMasks13,
    }
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(0.5)


#    # testing MobileNet DetectionNetwork
#    detectionNetwork = pipeline.createMobileNetDetectionNetwork()
#    detectionNetwork.setConfidenceThreshold(0.5)
#    detectionNetwork.setBlobPath(nnPath)

    # set up color camera and link to NN node
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(416, 416)
#    colorCam.setPreviewSize(300, 300)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setCamId(0)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.preview.link(detectionNetwork.input)


    # set up SPI out node
    spiOut = pipeline.createSPIOut()
    spiOut.setStreamName("spimetaout")
    spiOut.setBusId(0)
    detectionNetwork.out.link(spiOut.input)

    return pipeline


def test_pipeline():
    nnBlobPath="tiny-yolo-v3.blob.sh4cmx4NCE1"
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
