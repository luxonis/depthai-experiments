import cv2
import depthai as dai
from enum import Enum
import numpy as np
import numba as nb

EXP_STEP = 500
ISO_STEP = 50
LENS_STEP = 1
WB_STEP = 100

# Limits for manual controls
LENS_MIN = 0
LENS_MAX = 255
EXP_MIN = 1
# EXP_MAX is determined based on FPS
SENS_MIN = 100
SENS_MAX = 1600
WB_MIN = 1000
WB_MAX = 12000

class AwbMode(Enum):
    OFF = dai.CameraControl.AutoWhiteBalanceMode.OFF
    AUTO = dai.CameraControl.AutoWhiteBalanceMode.AUTO
    INCANDESCENT = dai.CameraControl.AutoWhiteBalanceMode.INCANDESCENT
    FLUORESCENT = dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT
    WARM_FLUORESCENT = dai.CameraControl.AutoWhiteBalanceMode.WARM_FLUORESCENT
    DAYLIGHT = dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT
    CLOUDY_DAYLIGHT = dai.CameraControl.AutoWhiteBalanceMode.CLOUDY_DAYLIGHT
    TWILIGHT = dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT
    SHADE = dai.CameraControl.AutoWhiteBalanceMode.SHADE

class AntiBandingMode(Enum):
    OFF = dai.CameraControl.AntiBandingMode.OFF
    MAINS_50_HZ = dai.CameraControl.AntiBandingMode.MAINS_50_HZ
    MAINS_60_HZ = dai.CameraControl.AntiBandingMode.MAINS_60_HZ
    AUTO = dai.CameraControl.AntiBandingMode.AUTO

class EffectMode(Enum):
    OFF = dai.CameraControl.EffectMode.OFF
    MONO = dai.CameraControl.EffectMode.MONO
    NEGATIVE = dai.CameraControl.EffectMode.NEGATIVE
    SOLARIZE = dai.CameraControl.EffectMode.SOLARIZE
    SEPIA = dai.CameraControl.EffectMode.SEPIA
    POSTERIZE = dai.CameraControl.EffectMode.POSTERIZE
    WHITEBOARD = dai.CameraControl.EffectMode.WHITEBOARD
    BLACKBOARD = dai.CameraControl.EffectMode.BLACKBOARD
    AQUA = dai.CameraControl.EffectMode.AQUA

class OtherControls(Enum):
    AWB_MODE, \
    AE_COMPENSATION, \
    ANTI_BANDING_MODE, \
    EFFECT_MODE, \
    BRIGHTNESS, \
    CONTRAST, \
    SATURATION = [ord(str(x)) for x in range(3, 10)]
    SHARPNESS = ord('0')
    LUMA_DENOISE = ord(']')
    CHROMA_DENOISE = ord('[')

# Packing scheme for RAW10 - MIPI CSI-2
# - 4 pixels: p0[9:0], p1[9:0], p2[9:0], p3[9:0]
# - stored on 5 bytes (byte0..4) as:
# | byte0[7:0] | byte1[7:0] | byte2[7:0] | byte3[7:0] |          byte4[7:0]             |
# |    p0[9:2] |    p1[9:2] |    p2[9:2] |    p3[9:2] | p3[1:0],p2[1:0],p1[1:0],p0[1:0] |
#
# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

    # for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

def clamp(num, v0, v1): return max(v0, min(num, v1))

class ManualCameraControl(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._lens_pos = 130
        self._exp_time = 20000
        self._sens_iso = 800
        self._wb_manual = 4000
        self._awb_mode_idx = -1
        self._anti_banding_mode_idx = -1
        self._effect_mode_idx = -1
        self._ae_comp = 0
        self._ae_lock = False
        self._awb_lock = False
        self._saturation = 0
        self._contrast = 0
        self._brightness = 0
        self._sharpness = 0
        self._luma_denoise = 0
        self._chroma_denoise = 0
        self._other_control_idx = None
        self._capture = False

    def build(self, preview_isp: dai.Node.Output, preview_raw: dai.Node.Output
              , control_queue: dai.Node.Input, fps: int, enable_raw: bool, focus_name: str) -> "ColorIspRaw":
        self.link_args(preview_isp, preview_raw)
        self.sendProcessingToPipeline(True)
        self.control_queue = control_queue
        # Need to reduce camera FPS (see .setFps) to be able to set higher exposure time
        self._exp_max = int(0.99 * 1000000 / fps)
        self._enable_raw = enable_raw
        self._focus_name = focus_name
        return self

    def process(self, preview_isp: dai.ImgFrame, preview_raw: dai.ImgFrame) -> None:
        cv_isp = self.update_output(preview_isp, False)
        cv_isp = self.show_controls(cv_isp)
        cv2.imshow("isp", cv_isp)
        if self._enable_raw:
            cv_raw = self.update_output(preview_raw, True)
            cv2.imshow("raw", cv_raw)

        self._capture = False
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()

        ctrl = dai.CameraControl()
        if key == ord('c'):
            self._capture = True
        elif key == ord('t'):
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            self._focus_name = "af"
            print("Autofocus trigger (and disable continuous)")
        elif key == ord('f'):
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            self._focus_name = 'af'
            print("Autofocus enable, continuous")
        elif key == ord('e'):
            ctrl.setAutoExposureEnable()
            print("Autoexposure enable")
        elif key in (ord(','), ord('.')):
            if key == ord(','): self._lens_pos -= LENS_STEP
            if key == ord('.'): self._lens_pos += LENS_STEP
            self._lens_pos = clamp(self._lens_pos, LENS_MIN, LENS_MAX)
            self._focus_name = 'f' + str(self._lens_pos)
            ctrl.setManualFocus(self._lens_pos)
            print("Setting manual focus, lens position:", self._lens_pos)
        elif key in (ord('i'), ord('o'), ord('k'), ord('l')):
            if key == ord('i'): self._exp_time -= EXP_STEP
            if key == ord('o'): self._exp_time += EXP_STEP
            if key == ord('k'): self._sens_iso -= ISO_STEP
            if key == ord('l'): self._sens_iso += ISO_STEP
            self._exp_time = clamp(self._exp_time, EXP_MIN, self._exp_max)
            self._sens_iso = clamp(self._sens_iso, SENS_MIN, SENS_MAX)
            ctrl.setManualExposure(self._exp_time, self._sens_iso)
            print("Setting manual exposure, time:", self._exp_time, "iso:", self._sens_iso)
        elif key in (ord('n'), ord('m')):
            if key == ord('n'): self._wb_manual -= WB_STEP
            if key == ord('m'): self._wb_manual += WB_STEP
            self._wb_manual = clamp(self._wb_manual, WB_MIN, WB_MAX)
            ctrl.setManualWhiteBalance(self._wb_manual)
            print("Setting manual white balance, temperature: ", self._wb_manual, "K")
        elif key == ord('1'):
            self._awb_lock = not self._awb_lock
            ctrl.setAutoWhiteBalanceLock(self._awb_lock)
            print("Auto white balance lock:", self._awb_lock)
        elif key == ord('2'):
            self._ae_lock = not self._ae_lock
            ctrl.setAutoExposureLock(self._ae_lock)
            print("Auto exposure lock:", self._ae_lock)
        elif key >= 0 and chr(key) in "34567890[]":
            self._other_control_idx = key
            print("Selected control:", OtherControls(key).name)
        elif key in (ord('-'), ord('_'), ord('+'), ord('=')):
            if key in (ord('-'), ord('_')): change = -1
            if key in (ord('+'), ord('=')): change = 1
            if self._other_control_idx is None:
                print("Please select a control first using keys 3..9, 0, [ ]")
            elif OtherControls(self._other_control_idx) == OtherControls.AWB_MODE:
                count = len(AwbMode)
                self._awb_mode_idx = (self._awb_mode_idx + change) % count
                awb_mode = list(AwbMode)[self._awb_mode_idx].value
                ctrl.setAutoWhiteBalanceMode(awb_mode)
                print("Auto white balance mode:", awb_mode)
            elif OtherControls(self._other_control_idx) == OtherControls.AE_COMPENSATION:
                self._ae_comp = clamp(self._ae_comp + change, -9, 9)
                ctrl.setAutoExposureCompensation(self._ae_comp)
                print("Auto exposure compensation:", self._ae_comp)
            elif OtherControls(self._other_control_idx) == OtherControls.ANTI_BANDING_MODE:
                count = len(AntiBandingMode)
                self._anti_banding_mode_idx = (self._anti_banding_mode_idx + change) % count
                anti_banding_mode = list(AntiBandingMode)[self._anti_banding_mode_idx].value
                ctrl.setAntiBandingMode(anti_banding_mode)
                print("Anti-banding mode:", anti_banding_mode)
            elif OtherControls(self._other_control_idx) == OtherControls.EFFECT_MODE:
                count = len(EffectMode)
                self._effect_mode_idx = (self._effect_mode_idx + change) % count
                effect_mode = list(EffectMode)[self._effect_mode_idx].value
                ctrl.setEffectMode(effect_mode)
                print("Effect mode:", effect_mode)
            elif OtherControls(self._other_control_idx) == OtherControls.BRIGHTNESS:
                self._brightness = clamp(self._brightness + change, -10, 10)
                ctrl.setBrightness(self._brightness)
                print("Brightness:", self._brightness)
            elif OtherControls(self._other_control_idx) == OtherControls.CONTRAST:
                self._contrast = clamp(self._contrast + change, -10, 10)
                ctrl.setContrast(self._contrast)
                print("Contrast:", self._contrast)
            elif OtherControls(self._other_control_idx) == OtherControls.SATURATION:
                self._saturation = clamp(self._saturation + change, -10, 10)
                ctrl.setSaturation(self._saturation)
                print("Saturation:", self._saturation)
            elif OtherControls(self._other_control_idx) == OtherControls.SHARPNESS:
                self._sharpness = clamp(self._sharpness + change, 0, 4)
                ctrl.setSharpness(self._sharpness)
                print("Sharpness:", self._sharpness)
            elif OtherControls(self._other_control_idx) == OtherControls.LUMA_DENOISE:
                self._luma_denoise = clamp(self._luma_denoise + change, 0, 4)
                ctrl.setLumaDenoise(self._luma_denoise)
                print("Luma denoise:", self._luma_denoise)
            elif OtherControls(self._other_control_idx) == OtherControls.CHROMA_DENOISE:
                self._chroma_denoise = clamp(self._chroma_denoise + change, 0, 4)
                ctrl.setChromaDenoise(self._chroma_denoise)
                print("Chroma denoise:", self._chroma_denoise)
        self.control_queue.send(ctrl)

    def update_output(self, frame: dai.ImgFrame, raw: bool) -> np.ndarray:
        width, height = frame.getWidth(), frame.getHeight()
        data = frame.getData()

        if self._capture:
            capture_file_info = "capture_{}_{}x{}_{}_{}".format(
                "raw" if raw else "isp",
                str(width),
                str(height),
                self._focus_name,
                str(frame.getSequenceNum())
            )

        if raw:
            pixel_order = cv2.COLOR_BayerBG2BGR
            raw_packed = data
            # Specific trim of leading metadata for IMX283
            if data.size == 5312 * (3692 + 18) * 5 // 4 or data.size == 3840 * (2160 + 22) * 5 // 4:
                if data.size == 5312 * (3692 + 18) * 5 // 4:
                    extra_offset = 64
                    raw_packed = data[(5312 * 18 + extra_offset) * 5 // 4:]
                    missing_data = np.full(extra_offset * 5 // 4, 0xAA, dtype=np.uint8)
                    raw_packed = np.append(raw_packed, missing_data)
                else:
                    raw_packed = data[3840 * 22 * 5 // 4:]
                pixel_order = cv2.COLOR_BayerGR2BGR
            unpacked = np.empty(raw_packed.size * 4 // 5, dtype=np.uint16)

            if self._capture:
                # Save to capture file on bits [9:0] of the 16-bit pixels
                unpack_raw10(raw_packed, unpacked, expand16bit=False)
                filename = capture_file_info + "_10bit.bw"
                unpacked.tofile(filename)
                print("Saving to file:", filename)
            # Full range for display, use bits [15:6] of the 16-bit pixels
            unpack_raw10(raw_packed, unpacked, expand16bit=True)
            shape = (height, width)
            bayer = unpacked.reshape(shape).astype(np.uint16)
            # See this for the ordering, at the end of page:
            # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
            bgr = cv2.cvtColor(bayer, pixel_order)

        else:
            if self._capture:
                filename = capture_file_info + "_P420.yuv"
                data.tofile(filename)
                print("Saving to file:", filename)
            shape = (height * 3 // 2, width)
            yuv420p = data.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

        bgr = np.ascontiguousarray(bgr)
        if self._capture:
            filename = capture_file_info + ".png"
            cv2.imwrite(filename, bgr)
            print("Saving to file:", filename)

        return bgr

    def show_controls(self, preview: dai.ImgFrame) -> dai.ImgFrame:
        controls = "User controls\n" \
            "'c' - to capture a set of images (from isp and/or raw streams)\n" \
            "'t' - to trigger autofocus\n" \
            "'ioklnm,.' for manual exposure/focus:\n" \
            "  Control:         key[dec/inc]  min..max\n" \
            "  exposure time:     i   o      1..33000 [us]\n" \
            "  sensitivity iso:      k   l      100..1600\n" \
            "  focus:              ,   .      0..255 [far..near]\n" \
            "  white balance:     n   m     1000..12000 (light color temperature K)\n\n" \
            "To go back to auto controls:\n" \
            "  'e' - autoexposure\n" \
            "  'f' - autofocus (continuous)\n\n" \
            "Other controls:\n" \
            "'1' - AWB lock (true / false)\n" \
            "'2' - AE lock (true / false)\n" \
            "'3' - Select control: AWB mode\n" \
            "'4' - Select control: AE compensation\n" \
            "'5' - Select control: anti-banding/flicker mode\n" \
            "'6' - Select control: effect mode\n" \
            "'7' - Select control: brightness\n" \
            "'8' - Select control: contrast\n" \
            "'9' - Select control: saturation\n" \
            "'0' - Select control: sharpness\n" \
            "'[' - Select control: luma denoise\n" \
            "']' - Select control: chroma denoise\n\n" \
            "For the 'Select control: ...' options, use these keys to modify the value:\n" \
            "  '-' or '_' to decrease\n" \
            "  '+' or '=' to increase\n".split("\n")

        for i, text in enumerate(controls):
            cv2.putText(preview, text, (20, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX
                        , 1, (255, 255, 255), 2, cv2.LINE_AA)
        return preview
