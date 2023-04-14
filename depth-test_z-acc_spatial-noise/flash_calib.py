#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('calib', nargs='?', help="Path to calibration file in json")
args = parser.parse_args()

# Connect device
with dai.Device(dai.OpenVINO.VERSION_2021_4, dai.UsbSpeed.HIGH) as device:

    deviceCalib = device.readCalibration()
    deviceCalib.eepromToJsonFile("backupCalib.json")
    print("Calibration Data on the device is backed up at:")
    print("backupCalib.json")
    calibData = dai.CalibrationHandler(args.calib)

    try:
        device.flashCalibration2(calibData)
        print('Successfully flashed calibration')
    except Exception as ex:
        print(f'Failed flashing calibration: {ex}')
