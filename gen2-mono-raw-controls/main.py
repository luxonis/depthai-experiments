#!/usr/bin/env python3

''' Control keys:
'IOKL' for manual exposure/iso:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O     20..33000 [us]
  sensitivity iso:   K   L    100..1600
To go back to auto controls:
  'E' - autoexposure
Other camera controls:
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
For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase

'C' - capture a set of frames (PNG and unprocessed)
'Q' - quit
'''

import cv2
import numpy as np
import numba as nb
import depthai as dai

streams = []
# Enable one or both streams
streams.append('isp')
streams.append('raw')

''' Packing scheme for RAW10 - MIPI CSI-2
- 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
- stored on 5 bytes (byte0..4) as:
| byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
|    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |
'''
# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

   #for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai:", dai.__version__, dai.__commit_datetime__)
pipeline = dai.Pipeline()

cam = pipeline.createMonoCamera()
cam.setBoardSocket(dai.CameraBoardSocket.LEFT)  # or RIGHT
cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
# Uncomment to be able to set a larger manual exposure, e.g: 10fps / 100ms
# cam.setFps(10)

# Camera control input
control = pipeline.createXLinkIn()
control.setStreamName('control')
control.out.link(cam.inputControl)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.out.link(xout_isp.input)

if 'raw' in streams:
    xout_raw = pipeline.createXLinkOut()
    xout_raw.setStreamName('raw')
    cam.raw.link(xout_raw.input)

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=False)
    q_list.append(q)
    # Make window resizable, and configure initial size
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (1280, 800))

controlQueue = device.getInputQueue('control')

# Manual exposure set step, configurable
EXP_STEP = 500  # us
ISO_STEP = 50

# Defaults and limits for manual focus/exposure controls

exp_time = 20000
exp_min = 20
# Note: need to reduce FPS (see .setFps) to be able to set higher exposure time
exp_max = 33000 # 1000000

sens_iso = 800
sens_min = 100
sens_max = 1600

# TODO make automatically iterable
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
                                 + '_' + str(data.getSequenceNum())
                                )
        if name == 'isp':
            if capture_flag:
                filename = capture_file_info_str + '_P400.yuv'
                print("Saving to file:", filename)
                payload.tofile(filename)
            shape = (height, width)
            bgr = payload.reshape(shape).astype(np.uint8)
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
            bgr = unpacked.reshape(shape).astype(np.uint16)
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
    elif key == ord('e'):
        print("Autoexposure enable")
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        controlQueue.send(ctrl)
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
        if   key == ord('3'): control = 'awb_mode'
        elif key == ord('4'): control = 'ae_comp'
        elif key == ord('5'): control = 'anti_banding_mode'
        elif key == ord('6'): control = 'effect_mode'
        elif key == ord('7'): control = 'brightness'
        elif key == ord('8'): control = 'contrast'
        elif key == ord('9'): control = 'saturation'
        elif key == ord('0'): control = 'sharpness'
        elif key == ord('['): control = 'luma_denoise'
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
        controlQueue.send(ctrl)
    elif key == ord('q'):
        break
