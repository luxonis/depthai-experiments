import cv2
import numpy as np
import argparse
import depthai as dai
import collections
import time
from itertools import cycle

cam_socket_opts = {
    'color'  : dai.CameraBoardSocket.CAM_A,
    'left' : dai.CameraBoardSocket.CAM_B,
    'right': dai.CameraBoardSocket.CAM_C,
    'cama' : dai.CameraBoardSocket.CAM_A,
    'camb' : dai.CameraBoardSocket.CAM_B,
    'camc' : dai.CameraBoardSocket.CAM_C,
    'camd' : dai.CameraBoardSocket.CAM_D,
    'came' : dai.CameraBoardSocket.CAM_E,
    'camf' : dai.CameraBoardSocket.CAM_F,
}



mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
    1200: dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    720:  dai.ColorCameraProperties.SensorResolution.THE_720_P,
    800:  dai.ColorCameraProperties.SensorResolution.THE_800_P,
    1080: dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    1200: dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    4000: dai.ColorCameraProperties.SensorResolution.THE_4000X3000,
    3040:   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '5mp': dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
    '48mp': dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}

class Camera_creation(object):
    def __init__(self, device, FPS = 30) -> None:
        self.xout = {}
        self.cam = {}
        self.streams = []
        self.fps = FPS
        self.device = device

    def pipelineCreation(self):
        self.pipeline = dai.Pipeline()

        self.control = self.pipeline.createXLinkIn()
        self.control.setStreamName('control')

        self.xinTofConfig = self.pipeline.createXLinkIn()
        self.xinTofConfig.setStreamName('tofConfig')

        self.features = self.device.getConnectedCameraFeatures()
        for cam_info in self.features:
            print(cam_info)
            self.streams.append(cam_info.name)
            self.xout[cam_info.name] = self.pipeline.createXLinkOut()
            self.xout[cam_info.name].setStreamName(cam_info.name)
            if len(cam_info.supportedTypes) > 1:
               answer = input("For camera info please select supported type:")
            else:
                self.sensorType(cam_info.name,cam_info.height,cam_info.supportedTypes[0])
            self.cam[cam_info.name].setBoardSocket(cam_socket_opts[cam_info.name])
            self.control.out.link(self.cam[cam_info.name].inputControl)
            self.cam[cam_info.name].setFps(self.fps)
        return self.pipeline

    def sensorType(self, name, resolution, supportedTypes):
        if supportedTypes == dai.CameraSensorType.MONO:
            self.cam[name] = self.pipeline.createMonoCamera()
            self.cam[name].setResolution(mono_res_opts[resolution])
            self.cam[name].out.link(self.xout[name].input)
        elif supportedTypes == dai.CameraSensorType.COLOR:
            self.cam[name] = self.pipeline.createColorCamera()
            self.cam[name].setResolution(color_res_opts[resolution])
            self.cam[name].isp.link(self.xout[name].input)
        else:
            print(f"Camera type {supportedTypes} not found!")

class FPS:
    def __init__(self, window_size=30):
        self.dq = collections.deque(maxlen=window_size)
        self.fps = 0

    def update(self, timestamp=None):
        if timestamp == None: timestamp = time.monotonic()
        count = len(self.dq)
        try:
            if count > 0: self.fps = count / (timestamp - self.dq[0])
        except ZeroDivisionError:
            pass
        self.dq.append(timestamp)

    def get(self):
        return self.fps

class CameraResponse:
    def __init__(self, device, lens = [0,255,150,1], iso = [100,1600,800,50], exposure = [1,33000,20000,500],
                 dot = [0,1200,0,100], flood = [0,1500,0,100] ) -> None:
        self.controlQueue = device.getInputQueue('control')
        self.lensMin, self.lensMax, self.lensPos, self.LENS_STEP = lens
        self.sensMin, self.sensMax, self.sensIso, self.ISO_STEP = iso
        self.expMin, self.expMax, self.expTime, self.EXP_STEP = exposure
        self.dotIntensity, self.DOT_MAX, self.dotIntensity, self.DOT_STEP = dot
        self.floodIntensity, self.FLOOD_MAX, self.floodIntensity, self.FLOOD_STEP = flood
        self.device = device
        # Defaults and limits for manual focus/exposure controls

        self.awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
        self.anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
        self.effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

        self.ae_comp = 0
        self.ae_lock = False
        self.awb_lock = False
        self.saturation = 0
        self.contrast = 0
        self.brightness = 0
        self.sharpness = 0
        self.luma_denoise = 0
        self.chroma_denoise = 0
        self.control = 'none'

    def CameraControl(self, key, streams):
        if key == ord('c'):
            capture_list = streams.copy()
            capture_time = time.strftime('%Y%m%d_%H%M%S')
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            self.controlQueue.send(ctrl)
        elif key == ord('f'):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            self.controlQueue.send(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self.controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): self.lensPos -= self.LENS_STEP
            if key == ord('.'): self.lensPos += self.LENS_STEP
            self.lensPos = self.clamp(self.lensPos, self.lensMin, self.lensMax)
            print("Setting manual focus, lens position: ", self.lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(self.lensPos)
            self.controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): self.expTime -= self.EXP_STEP
            if key == ord('o'): self.expTime += self.EXP_STEP
            if key == ord('k'): self.sensIso -= self.ISO_STEP
            if key == ord('l'): self.sensIso += self.ISO_STEP
            self.expTime = self.clamp(self.expTime, self.expMin, self.expMax)
            self.sensIso = self.clamp(self.sensIso, self.sensMin, self.sensMax)
            print("Setting manual exposure, time: ", self.expTime, "iso: ", self.sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(self.expTime, self.sensIso)
            self.controlQueue.send(ctrl)
        elif key == ord('1'):
            self.awb_lock = not self.awb_lock
            print("Auto white balance lock:", self.awb_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(self.awb_lock)
            self.controlQueue.send(ctrl)
        elif key == ord('2'):
            self.ae_lock = not self.ae_lock
            print("Auto exposure lock:", self.ae_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(self.ae_lock)
            self.controlQueue.send(ctrl)
        elif key == ord('a'):
            self.dotIntensity = self.dotIntensity - self.DOT_STEP
            if self.dotIntensity < 0:
                self.dotIntensity = 0
            self.device.setIrLaserDotProjectorBrightness(self.dotIntensity)
        elif key == ord('d'):
            self.dotIntensity = self.dotIntensity + self.DOT_STEP
            if self.dotIntensity > self.DOT_MAX:
                self.dotIntensity = self.DOT_MAX
            self.device.setIrLaserDotProjectorBrightness(self.dotIntensity)
        elif key == ord('w'):
            self.floodIntensity = self.floodIntensity + self.FLOOD_STEP
            if self.floodIntensity > self.FLOOD_MAX:
                self.floodIntensity = self.FLOOD_MAX
            self.device.setIrFloodLightBrightness(self.floodIntensity)
        elif key == ord('s'):
            self.floodIntensity = self.floodIntensity - self.FLOOD_STEP
            if self.floodIntensity < 0:
                self.floodIntensity = 0
            self.device.setIrFloodLightBrightness(self.floodIntensity)
        elif key >= 0 and chr(key) in '34567890[]p':
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
            elif key == ord('p'):
                control = 'tof_amplitude_min'
            print("Selected control:", control)
        elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
            change = 0
            if key in [ord('-'), ord('_')]: change = -1
            if key in [ord('+'), ord('=')]: change = 1
            ctrl = dai.CameraControl()
            if self.control == 'none':
                print("Please select a control first using keys 3..9 0 [ ]")
            elif self.control == 'ae_comp':
                self.ae_comp = self.clamp(self.ae_comp + change, -9, 9)
                print("Auto exposure compensation:", self.ae_comp)
                ctrl.setAutoExposureCompensation(self.ae_comp)
            elif self.control == 'anti_banding_mode':
                self.abm = next(self.anti_banding_mode)
                print("Anti-banding mode:", self.abm)
                ctrl.setAntiBandingMode(self.abm)
            elif self.control == 'awb_mode':
                self.awb = next(self.awb_mode)
                print("Auto white balance mode:", self.awb)
                ctrl.setAutoWhiteBalanceMode(self.awb)
            elif self.control == 'effect_mode':
                self.eff = next(self.effect_mode)
                print("Effect mode:", self.eff)
                ctrl.setEffectMode(self.eff)
            elif self.control == 'brightness':
                self.brightness = self.clamp(self.brightness + change, -10, 10)
                print("Brightness:", self.brightness)
                ctrl.setBrightness(self.brightness)
            elif self.control == 'contrast':
                self.contrast = self.clamp(self.contrast + change, -10, 10)
                print("Contrast:", self.contrast)
                ctrl.setContrast(self.contrast)
            elif self.control == 'saturation':
                self.saturation = self.clamp(self.saturation + change, -10, 10)
                print("Saturation:", self.saturation)
                ctrl.setSaturation(self.saturation)
            elif self.control == 'sharpness':
                self.sharpness = self.clamp(self.sharpness + change, 0, 4)
                print("Sharpness:", self.sharpness)
                ctrl.setSharpness(self.sharpness)
            elif self.control == 'luma_denoise':
                self.luma_denoise = self.clamp(self.luma_denoise + change, 0, 4)
                print("Luma denoise:", self.luma_denoise)
                ctrl.setLumaDenoise(self.luma_denoise)
            elif self.control == 'chroma_denoise':
                self.chroma_denoise = self.clamp(self.chroma_denoise + change, 0, 4)
                print("Chroma denoise:", self.chroma_denoise)
                ctrl.setChromaDenoise(self.chroma_denoise)
            self.controlQueue.send(ctrl)

    def clamp(self,num, v0, v1):
     return max(v0, min(num, v1))