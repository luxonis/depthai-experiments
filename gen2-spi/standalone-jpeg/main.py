import cv2
import numpy as np
import depthai as dai
import sys
from time import sleep

'''
Basic demo of gen2 pipeline builder functionality where output of jpeg encoded images are sent out SPI rather than the typical XLink out interface.

Make sure you have something to handle the SPI protocol on the other end! See the included ESP32 example. 
'''

def create_spi_demo_pipeline():
    print("Creating SPI pipeline: ")
    print("COLOR CAM -> ENCODER -> SPI OUT")
    pipeline = dai.Pipeline()

    cam_color         = pipeline.createColorCamera()
    spiout_preview    = pipeline.createSPIOut()
    videnc            = pipeline.createVideoEncoder()

    # set up color camera and link to NN node
    cam_color.setPreviewSize(300, 300);
    cam_color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P);
    cam_color.setInterleaved(False);
    cam_color.setCamId(0);
    cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR);

    # VideoEncoder
    videnc.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.MJPEG);

    # Link plugins CAM -> ENCODER -> SPI OUT
    cam_color.video.link(videnc.input);
    spiout_preview.setStreamName("spipreview");
    spiout_preview.setBusId(0);
    videnc.bitstream.link(spiout_preview.input);

    return pipeline

def flash_bootloader():
    (f, bl) = dai.DeviceBootloader.getFirstAvailableDevice()
    bootloader = dai.DeviceBootloader(bl)
    print(bootloader.getVersion())

    progress = lambda p : print(f'Flashing progress: {p*100:.1f}%')
    bootloader.flashBootloader(progress)


def flash_image():
    pipeline = create_spi_demo_pipeline()
    
    (found, bl) = dai.DeviceBootloader.getFirstAvailableDevice()

    if(found):
        bootloader = dai.DeviceBootloader(bl)
        progress = lambda p : print(f'Flashing progress: {p*100:.1f}%')
        bootloader.flash(progress, pipeline)
    else:
        print("No booted (bootloader) devices found...")

def write_image_to_file(filename):
    pipeline = create_spi_demo_pipeline()
    dai.DeviceBootloader.saveDepthaiApplicationPackage(filename, pipeline)

if(len(sys.argv) >= 2 and sys.argv[1] == "bootloader"):
    print("flashing bootloader")
    flash_bootloader()
elif(len(sys.argv) >= 2 and sys.argv[1] == "save"):
    filename = "pipeline.dap"
    print("saving pipeline to disk as " + filename)
    write_image_to_file(filename)
else:
    print("flashing pipeline")
    flash_image()
