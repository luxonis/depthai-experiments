#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import depthai as dai
import numba as nb
import numpy as np

''' User controls
'C' - to capture a set of assets (from isp and/or raw streams)
'T' - to trigger autofocus
'IOKL,.' for manual exposure/focus:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
  white balance:     N   M   1000..12000 (light color temperature K)
To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)
Other controls:
'1' - AWB lock (true / false)
'2' - AE lock (true / false)
'3' - Select control: AWB mode
'4' - Select control: AE compensation
'5' - Select control: anti-banding/flicker mode
'6' - Select control: effect mode
'7' - Select control: brightness
'8' - Select control: contrast
'9' - Select control: saturation
'0' - Select control: sharpness
'[' - Select control: luma denoise
']' - Select control: chroma denoise

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase
'''

parser = argparse.ArgumentParser()
parser.add_argument('-res', '--resolution', default='1080', choices={'1080', '4k', '12mp', '13mp'},
                    help="Select RGB resolution. Default: %(default)s")
parser.add_argument('-raw', '--enable_raw', default=False, action="store_true",
                    help='Enable the color RAW stream')
parser.add_argument('-fps', '--fps', default=15, type=int,
                    help="Camera FPS. Default: %(default)s")
parser.add_argument('-lens', '--lens_position', default=-1, type=int,
                    help="Lens position for manual focus 0..255, or auto: -1. Default: %(default)s")
parser.add_argument('-ds', '--isp_downscale', default=1, type=int,
                    help="Downscale the ISP output by this factor")
parser.add_argument('-tun', '--camera_tuning', type=Path,
                    help="Path to custom camera tuning database")
parser.add_argument('-rot', '--rotate', action='store_true',
                    help="Camera image orientation set to 180 degrees rotation")

args = parser.parse_args()

streams = []
# Enable one or both streams
streams.append('isp')
if args.enable_raw:
    streams.append('raw')

''' Packing scheme for RAW10 - MIPI CSI-2
- 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
- stored on 5 bytes (byte0..4) as:
| byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
|    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |
'''


# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1](nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

    # for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5):  # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4] = ((input[i * 5] << 2) | (b4 & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) | (b4 >> 6)) << lShift

    return out


print("depthai version:", dai.__version__)

rgb_res_opts = {
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k': dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
}
rgb_res = rgb_res_opts.get(args.resolution)

pipeline = dai.Pipeline()

if args.camera_tuning:
    pipeline.setCameraTuningBlobPath(str(args.camera_tuning))

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(rgb_res)
# Optional, set manual focus. 255: macro (8cm), about 120..130: infinity
focus_name = 'af'
if args.lens_position >= 0:
    cam.initialControl.setManualFocus(args.lens_position)
    focus_name = 'f' + str(args.lens_position)
cam.setIspScale(1, args.isp_downscale)
cam.setFps(args.fps)  # Default: 30
if args.rotate:
    cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

# Camera control input
control = pipeline.create(dai.node.XLinkIn)
control.setStreamName('control')
control.out.link(cam.inputControl)

if 'isp' in streams:
    xout_isp = pipeline.create(dai.node.XLinkOut)
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

if 'raw' in streams:
    xout_raw = pipeline.create(dai.node.XLinkOut)
    xout_raw.setStreamName('raw')
    cam.raw.link(xout_raw.input)

device = dai.Device(pipeline)

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))

controlQueue = device.getInputQueue('control')

# Manual exposure/focus set step, configurable
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 1
WB_STEP = 100

# Defaults and limits for manual focus/exposure controls
lens_pos = 130
lens_min = 0
lens_max = 255

exp_time = 20000
exp_min = 1
# Note: need to reduce FPS (see .setFps) to be able to set higher exposure time
# With the custom FW, larger exposures can be set automatically (requirements not yet updated)
exp_max = int(0.99 * 1000000 / args.fps)

sens_iso = 800
sens_min = 100
sens_max = 1600

wb_manual = 4000
wb_min = 1000
wb_max = 12000

# TODO how can we make the enums automatically iterable?
awb_mode_idx = -1
awb_mode_list = [
    dai.CameraControl.AutoWhiteBalanceMode.OFF,
    dai.CameraControl.AutoWhiteBalanceMode.AUTO,
    dai.CameraControl.AutoWhiteBalanceMode.INCANDESCENT,
    dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT,
    dai.CameraControl.AutoWhiteBalanceMode.WARM_FLUORESCENT,
    dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT,
    dai.CameraControl.AutoWhiteBalanceMode.CLOUDY_DAYLIGHT,
    dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT,
    dai.CameraControl.AutoWhiteBalanceMode.SHADE,
]

anti_banding_mode_idx = -1
anti_banding_mode_list = [
    dai.CameraControl.AntiBandingMode.OFF,
    dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
    dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
    dai.CameraControl.AntiBandingMode.AUTO,
]

effect_mode_idx = -1
effect_mode_list = [
    dai.CameraControl.EffectMode.OFF,
    dai.CameraControl.EffectMode.MONO,
    dai.CameraControl.EffectMode.NEGATIVE,
    dai.CameraControl.EffectMode.SOLARIZE,
    dai.CameraControl.EffectMode.SEPIA,
    dai.CameraControl.EffectMode.POSTERIZE,
    dai.CameraControl.EffectMode.WHITEBOARD,
    dai.CameraControl.EffectMode.BLACKBOARD,
    dai.CameraControl.EffectMode.AQUA,
]

ae_comp = 0  # Valid: -9 .. +9
ae_lock = False
awb_lock = False
saturation = 0
contrast = 0
brightness = 0
sharpness = 0
luma_denoise = 0
chroma_denoise = 0
control = 'none'


def clamp(num, v0, v1): return max(v0, min(num, v1))


capture_flag = False
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        payload = data.getData()
        capture_file_info_str = ('capture_' + name
                                 + '_' + str(width) + 'x' + str(height)
                                 + '_' + focus_name
                                 + '_' + str(data.getSequenceNum())
                                 )
        if name == 'isp':
            if capture_flag:
                filename = capture_file_info_str + '_P420.yuv'
                print("Saving to file:", filename)
                payload.tofile(filename)
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
        if name == 'raw':
            # Preallocate the output buffer
            unpacked = np.empty(payload.size * 4 // 5, dtype=np.uint16)
            if capture_flag:
                # Save to capture file on bits [9:0] of the 16-bit pixels
                unpack_raw10(payload, unpacked, expand16bit=False)
                filename = capture_file_info_str + '_10bit.bw'
                print("Saving to file:", filename)
                unpacked.tofile(filename)
            # Full range for display, use bits [15:6] of the 16-bit pixels
            unpack_raw10(payload, unpacked, expand16bit=True)
            shape = (height, width)
            bayer = unpacked.reshape(shape).astype(np.uint16)
            # See this for the ordering, at the end of page:
            # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        if capture_flag:  # Save to disk if `c` was pressed
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            bgr = np.ascontiguousarray(bgr)  # just in case
            cv2.imwrite(filename, bgr)
        bgr = np.ascontiguousarray(bgr)  # just in case
        cv2.imshow(name, bgr)
    # Reset capture_flag after iterating through all streams
    capture_flag = False
    key = cv2.waitKey(1)
    if key == ord('c'):
        capture_flag = True
    elif key == ord('t'):
        print("Autofocus trigger (and disable continuous)")
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        controlQueue.send(ctrl)
        focus_name = 'af'
    elif key == ord('f'):
        print("Autofocus enable, continuous")
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        controlQueue.send(ctrl)
        focus_name = 'af'
    elif key == ord('e'):
        print("Autoexposure enable")
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        controlQueue.send(ctrl)
    elif key in [ord(','), ord('.')]:
        if key == ord(','): lens_pos -= LENS_STEP
        if key == ord('.'): lens_pos += LENS_STEP
        lens_pos = clamp(lens_pos, lens_min, lens_max)
        print("Setting manual focus, lens position:", lens_pos)
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(lens_pos)
        controlQueue.send(ctrl)
        focus_name = 'f' + str(lens_pos)
    elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
        if key == ord('i'): exp_time -= EXP_STEP
        if key == ord('o'): exp_time += EXP_STEP
        if key == ord('k'): sens_iso -= ISO_STEP
        if key == ord('l'): sens_iso += ISO_STEP
        exp_time = clamp(exp_time, exp_min, exp_max)
        sens_iso = clamp(sens_iso, sens_min, sens_max)
        print("Setting manual exposure, time:", exp_time, "iso:", sens_iso)
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exp_time, sens_iso)
        controlQueue.send(ctrl)
    elif key in [ord('n'), ord('m')]:
        if key == ord('n'): wb_manual -= WB_STEP
        if key == ord('m'): wb_manual += WB_STEP
        wb_manual = clamp(wb_manual, wb_min, wb_max)
        print("Setting manual white balance, temperature: ", wb_manual, "K")
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(wb_manual)
        controlQueue.send(ctrl)
    elif key == ord('1'):
        awb_lock = not awb_lock
        print("Auto white balance lock:", awb_lock)
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceLock(awb_lock)
        controlQueue.send(ctrl)
    elif key == ord('2'):
        ae_lock = not ae_lock
        print("Auto exposure lock:", ae_lock)
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureLock(ae_lock)
        controlQueue.send(ctrl)
    elif key >= 0 and chr(key) in '34567890[]':
        if key == ord('3'):
            control = 'awb_mode'
        elif key == ord('4'):
            control = 'ae_comp'
        elif key == ord('5'):
            control = 'anti_banding_mode'
        elif key == ord('6'):
            control = 'effect_mode'
        elif key == ord('7'):
            control = 'brightness'
        elif key == ord('8'):
            control = 'contrast'
        elif key == ord('9'):
            control = 'saturation'
        elif key == ord('0'):
            control = 'sharpness'
        elif key == ord('['):
            control = 'luma_denoise'
        elif key == ord(']'):
            control = 'chroma_denoise'
        print("Selected control:", control)
    elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
        change = 0
        if key in [ord('-'), ord('_')]: change = -1
        if key in [ord('+'), ord('=')]: change = 1
        ctrl = dai.CameraControl()
        if control == 'none':
            print("Please select a control first using keys 3..9 0 [ ]")
        elif control == 'ae_comp':
            ae_comp = clamp(ae_comp + change, -9, 9)
            print("Auto exposure compensation:", ae_comp)
            ctrl.setAutoExposureCompensation(ae_comp)
        elif control == 'anti_banding_mode':
            cnt = len(anti_banding_mode_list)
            anti_banding_mode_idx = (anti_banding_mode_idx + cnt + change) % cnt
            anti_banding_mode = anti_banding_mode_list[anti_banding_mode_idx]
            print("Anti-banding mode:", anti_banding_mode)
            ctrl.setAntiBandingMode(anti_banding_mode)
        elif control == 'awb_mode':
            cnt = len(awb_mode_list)
            awb_mode_idx = (awb_mode_idx + cnt + change) % cnt
            awb_mode = awb_mode_list[awb_mode_idx]
            print("Auto white balance mode:", awb_mode)
            ctrl.setAutoWhiteBalanceMode(awb_mode)
        elif control == 'effect_mode':
            cnt = len(effect_mode_list)
            effect_mode_idx = (effect_mode_idx + cnt + change) % cnt
            effect_mode = effect_mode_list[effect_mode_idx]
            print("Effect mode:", effect_mode)
            ctrl.setEffectMode(effect_mode)
        elif control == 'brightness':
            brightness = clamp(brightness + change, -10, 10)
            print("Brightness:", brightness)
            ctrl.setBrightness(brightness)
        elif control == 'contrast':
            contrast = clamp(contrast + change, -10, 10)
            print("Contrast:", contrast)
            ctrl.setContrast(contrast)
        elif control == 'saturation':
            saturation = clamp(saturation + change, -10, 10)
            print("Saturation:", saturation)
            ctrl.setSaturation(saturation)
        elif control == 'sharpness':
            sharpness = clamp(sharpness + change, 0, 4)
            print("Sharpness:", sharpness)
            ctrl.setSharpness(sharpness)
        elif control == 'luma_denoise':
            luma_denoise = clamp(luma_denoise + change, 0, 4)
            print("Luma denoise:", luma_denoise)
            ctrl.setLumaDenoise(luma_denoise)
        elif control == 'chroma_denoise':
            chroma_denoise = clamp(chroma_denoise + change, 0, 4)
            print("Chroma denoise:", chroma_denoise)
            ctrl.setChromaDenoise(chroma_denoise)
        controlQueue.send(ctrl)
    elif key == ord('q'):
        break
