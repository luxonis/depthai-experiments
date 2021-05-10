#!/usr/bin/env python3

import argparse
import depthai as dai

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f',  '--flash-usb-boot', action='store_true')
group.add_argument('-fb', '--flash-bootloader', action='store_true')
group.add_argument('-s',  '--switch-to-usb-boot', action='store_true')
args = parser.parse_args()

print('DepthAI version:', dai.__version__)

(f, bl) = dai.DeviceBootloader.getFirstAvailableDevice()
bootloader = dai.DeviceBootloader(bl)

if args.flash_usb_boot or args.flash_bootloader:
    # Workaround for a bug with the timeout-enabled bootloader
    progressCalled = False

    def progress(p):
        global progressCalled
        progressCalled = True
        print(f'Flashing progress: {p*100:.1f}%')

    if args.flash_usb_boot:
        print('Flashing USB boot header...')
        bootloader.flashUsbBoot(progress)
    elif args.flash_bootloader:
        print('Flashing bootloader...')
        bootloader.flashBootloader(progress)

    if not progressCalled:
        raise RuntimeError('Flashing failed, please try again')
    print('Done.')

if args.switch_to_usb_boot:
    print('Switching to USB boot')
    bootloader.switchToUsbBoot()
