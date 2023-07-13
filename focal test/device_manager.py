#!/usr/bin/env python3

USE_OPENCV = False

from datetime import timedelta
import depthai as dai
import tempfile
import PySimpleGUI as sg
import sys
from typing import Dict
import platform
import os

if USE_OPENCV:
    # import cv2
    pass
else:
    import io
    from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if platform.system() == 'Windows':
    sg.set_global_icon(f'{SCRIPT_DIR}/assets/icon.ico')
else:
    sg.set_global_icon(f'{SCRIPT_DIR}/assets/icon.png')

CONF_TEXT_POE = ['ipTypeText', 'ipText', 'maskText', 'gatewayText', 'dnsText', 'dnsAltText', 'networkTimeoutText', 'macText']
CONF_INPUT_POE = ['staticBut', 'dynamicBut', 'ip', 'mask', 'gateway', 'dns', 'dnsAlt', 'networkTimeout', 'mac']
CONF_TEXT_USB = ['usbTimeoutText', 'usbSpeedText']
CONF_INPUT_USB = ['usbTimeout', 'usbSpeed']
USB_SPEEDS = ["UNKNOWN", "LOW", "FULL", "HIGH", "SUPER", "SUPER_PLUS"]

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    print('Exception in {}, line {}; {}'.format(filename, lineno, exc_obj))

def check_ip(s: str, req = True):
    if s == "":
        return not req
    spl = s.split(".")
    if len(spl) != 4:
        sg.Popup("Wrong IP format.\nValue should be similar to 255.255.255.255")
        return False
    for num in spl:
        if 255 < int(num):
            sg.Popup("Wrong IP format.\nValues can not be above 255!")
            return False
    return True

def check_mac(s):
    if s.count(":") != 5:
        sg.Popup("Wrong MAC format.\nValue should be similar to FF:FF:FF:FF:FF:FF")
        return False
    for i in s.split(":"):
        for j in i:
            if j > "F" or (j < "A" and not j.isdigit()) or len(i) != 2:
                sg.Popup("Wrong MAC format.\nValue should be similar to FF:FF:FF:FF:FF:FF")
                return False
    return True

class Progress:
    def __init__(self, txt = 'Flashing progress: 0.0%'):
        layout = [
            [sg.Text(txt, key='txt')],
            [sg.ProgressBar(1.0, orientation='h', size=(20,20), key='progress')],
        ]
        self.window = sg.Window("Progress", layout, modal=True, finalize=True)
    def update(self, val):
        self.window['progress'].update(val)
        self.window['txt'].update(f'Flashing progress: {val*100:.1f}%')
    def finish(self, msg):
        self.window.close()
        sg.Popup(msg)

class SelectBootloader:
    def __init__(self, options, text):
        self.ok = False
        layout = [
            [sg.Text(text)],
            [
                sg.Text("Bootloader type:", font=('Arial', 10, 'bold'), text_color="black"),
                sg.Combo(options, options[0], size=(30, 3), key="bootType"),
            ],
            [sg.Text("Warning! This may soft-brick your device,\nproceed if you are sure what you are doing!")],
            [sg.Submit(), sg.Cancel()],
        ]
        self.window = sg.Window("Select bootloader", layout, size=(300,150), modal=True, finalize=True)
    def wait(self):
        event, values = self.window.Read()
        self.window.close()
        if values is not None:
            type = getattr(dai.DeviceBootloader.Type, values['bootType'])
            return (str(event) == "Submit", type)
        else:
            return (False, None)

class AreYouSure:
    def __init__(self, text):
        self.ok = False
        layout = [
            [sg.Text(text)],
            [sg.Submit(button_text="Yes"), sg.Cancel(button_text="No")],
        ]
        self.window = sg.Window("Are You Sure?", layout, size=(450,150), modal=True, finalize=True)
    def wait(self):
        event, values = self.window.Read()
        self.window.close()
        if values is not None:
            return str(event) == "Submit"
        else:
            return False


class SelectIP:
    def __init__(self):
        self.ok = False
        layout = [
            [sg.Text("Specify the custom IP of the OAK PoE\ncamera you want to connect to")],
            [
                sg.Text("IPv4:", font=('Arial', 10, 'bold'), text_color="black"),
                sg.InputText(key="ip", size=(16, 2)),
            ],
            [sg.Submit(), sg.Cancel()],
        ]
        self.window = sg.Window("Specify IP", layout, size=(300,110), modal=True, finalize=True)
    def wait(self):
        event, values = self.window.Read()
        self.window.close()
        if str(event) == "Cancel" or values is None or not check_ip(values["ip"]):
            return False, ""
        return True, values["ip"]

class SearchDevice:
    def __init__(self):
        self.infos = []
        layout = [
            [sg.Text("Select an OAK camera you would like to connect to.", font=('Arial', 10, 'bold'))],
            [sg.Table(values=[], headings=["MxId", "Name", "State"],
                # col_widths=[25, 17, 17],
                # def_col_width=25,
                # max_col_width=200,
                # background_color='light blue',
                display_row_numbers=True,
                justification='right',
                num_rows=8,
                alternating_row_color='lightyellow',
                key='table',
                row_height=35,
                # size=(500, 300)
                expand_x = True,
                enable_events = True,
                enable_click_events = True,
                )
            ],
            [sg.Button('Search', size=(15, 2), font=('Arial', 10, 'bold'))],
        ]
        self.window = sg.Window("Select Device", layout, size=(550,375), modal=True, finalize=True)
        self.search_devices()

    def search_devices(self):
        self.infos = dai.XLinkConnection.getAllConnectedDevices()
        if not self.infos:
            sg.Popup("No devices found.")
        else:
            rows = []
            for info in self.infos:
                rows.append([info.getMxId(), info.name, deviceStateTxt(info.state)])
            self.window['table'].update(values=rows)

    def wait(self) -> dai.DeviceInfo:
        while True:
            event, values = self.window.Read()
            if event is None:
                self.window.close()
                return None
            elif str(event) == 'Search':
                self.search_devices()
            elif len(event) == 3 and event[0] == "table" and event[1] == "+CLICKED+":
                # User selected a device
                deviceIndex = event[2][0]
                if deviceIndex is not None:
                    deviceSelected = self.infos[deviceIndex]
                    self.window.close()
                    return deviceSelected

def flashBootloader(bl: dai.DeviceBootloader, device: dai.DeviceInfo, type: dai.DeviceBootloader.Type):
    factoryBlWarningMessage = """Main Bootloader type or version doesn't support User Bootloader flashing.
Main (factory) bootloader will be updated instead.
Proceed with caution
    """

    try:
        if bl.isUserBootloaderSupported():
            pr = Progress('Flashing...')
            progress = lambda p : pr.update(p)
            bl.flashUserBootloader(progress)
            pr.finish("Flashed newest User Bootloader version.")
        elif AreYouSure(text=factoryBlWarningMessage).wait():
            bl.close()
            pr = Progress('Connecting...')
            bl = dai.DeviceBootloader(device, True)
            progress = lambda p : pr.update(p)
            if type == dai.DeviceBootloader.Type.AUTO:
                type = bl.getType()
            bl.flashBootloader(memory=dai.DeviceBootloader.Memory.FLASH, type=type, progressCallback=progress)
            pr.finish("Flashed newest Factory Bootloader version.")
        else:
            return False
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')

    return True


# Danger
def flashFactoryBootloader(device: dai.DeviceInfo, type: dai.DeviceBootloader.Type):
    try:
        pr = Progress('Connecting...')
        bl = dai.DeviceBootloader(device, True)
        progress = lambda p : pr.update(p)
        if type == dai.DeviceBootloader.Type.AUTO:
            type = bl.getType()
        bl.flashBootloader(memory=dai.DeviceBootloader.Memory.FLASH, type=type, progressCallback=progress)
        pr.finish("Flashed newest bootloader version.")
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')

def factoryReset(device: dai.DeviceInfo, type: dai.DeviceBootloader.Type):
    try:
        pr = Progress('Preparing and connecting...')

        blBinary = dai.DeviceBootloader.getEmbeddedBootloaderBinary(type)
        # Clear 1 MiB for USB BL and 8 MiB for NETWORK BL
        mib = 1 if type == dai.DeviceBootloader.Type.USB else 8
        blBinary = blBinary + ([0xFF] * ((mib * 1024 * 1024 + 512) - len(blBinary)))
        tmpBlFw = tempfile.NamedTemporaryFile(delete=False)
        tmpBlFw.write(bytes(blBinary))

        bl = dai.DeviceBootloader(device, True)

        progress = lambda p : pr.update(p)
        success, msg = bl.flashBootloader(progress, tmpBlFw.name)
        msg = "Factory reset was successful." if success else f"Factory reset failed. Error: {msg}"
        pr.finish(msg)
        tmpBlFw.close()
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')

def connectAndStartStreaming(dev):

    # OpenCV
    if USE_OPENCV:
        # Create pipeline
        pipeline = dai.Pipeline()

        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setIspScale(1,3)
        videnc = pipeline.create(dai.node.VideoEncoder)
        videnc.setDefaultProfilePreset(camRgb.getFps(), videnc.Properties.Profile.MJPEG)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("mjpeg")
        camRgb.video.link(videnc.input)
        videnc.bitstream.link(xout.input)

        with dai.Device(pipeline, dev) as d:
            while not d.isClosed():
                mjpeg = d.getOutputQueue('mjpeg').get()
                frame = cv2.imdecode(mjpeg.getData(), cv2.IMREAD_UNCHANGED)
                cv2.imshow('Color Camera', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyWindow('Color Camera')
                    break
    else:
        # Create pipeline (no opencv)
        pipeline = dai.Pipeline()
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setIspScale(1,3)
        camRgb.setPreviewSize(camRgb.getIspSize())
        camRgb.setColorOrder(camRgb.Properties.ColorOrder.RGB)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.input.setQueueSize(2)
        xout.input.setBlocking(False)
        xout.setStreamName("color")
        camRgb.preview.link(xout.input)

        with dai.Device(pipeline, dev) as d:
            frame = d.getOutputQueue('color', 2, False).get()
            width, height = frame.getWidth(), frame.getHeight()

            layout = [[sg.Graph(
                canvas_size=(width, height),
                graph_bottom_left=(0, 0),
                graph_top_right=(width, height),
                key="-GRAPH-",
                change_submits=True,  # mouse click events
                background_color='lightblue',
                drag_submits=True), ],]
            window = sg.Window("Color Camera Stream", layout, finalize=True)
            graph = window["-GRAPH-"]

            while not d.isClosed():
                frame = d.getOutputQueue('color').get()
                with io.BytesIO() as output:
                    rgb = frame.getFrame()
                    image = Image.fromarray(rgb, "RGB")
                    image.save(output, format="GIF")
                    contents = output.getvalue()
                    graph.draw_image(data=contents, location=(0, height))

                event, values = window.read(timeout=1)
                if event == sg.WIN_CLOSED:
                    break
            window.close()


def flashFromFile(file, bl: dai.DeviceBootloader):
    try:
        if str(file)[-3:] == "dap":
            pr = Progress('Flashing application...')
            progress = lambda p : pr.update(p)
            with open(file, mode = 'rb') as f:
                dap = list(f.read())
                success, msg = bl.flashDepthaiApplicationPackage(progress, dap)
                msg = "Flashing application was successful." if success else f"Flashing application failed. Error: {msg}"
                pr.finish(msg)
        else:
            sg.Popup("Selected file is not .dap!")
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')

def recoveryMode(bl: dai.DeviceBootloader):
    try:
        bl.bootUsbRomBootloader()
        return True
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')
    return False

def connectToDevice(device: dai.DeviceInfo) -> dai.DeviceBootloader:
    try:
        bl = dai.DeviceBootloader(device, False)
        return bl
    except Exception as ex:
        PrintException()
        sg.Popup(f'{ex}')

def deviceStateTxt(state: dai.XLinkDeviceState) -> str:
    return str(state).replace("XLinkDeviceState.X_LINK_", "")

# Start defying layout

sg.theme('LightGrey2')

# layout for device tab
aboutDeviceLayout = [
    [sg.Text("About device", size=(30, 1), font=('Arial', 30, 'bold'), text_color="black")],
    [sg.HSeparator()],
    [
        sg.Button("About device", size=(15, 1), font=('Arial', 10, 'bold'), disabled=True, key="_unique_aboutBtn"),
        sg.Button("Config", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_configBtn"),
        sg.Button("Application", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_appBtn"),
        sg.Button("Danger Zone", size=(15, 1), font=('Arial', 10, 'bold'), button_color='#FF0500', disabled=False,  key="_unique_dangerBtn"),
    ],
    [sg.HSeparator()],
    [
        sg.Text("Select device: ", size=(15, 1), font=('Arial', 10, 'bold'), text_color="black"),
        sg.Combo([], "Select device", size=(30, 5), key="devices", enable_events=True),
        sg.Button("Search", font=('Arial', 10, 'bold')),
        sg.Button("Specify IP")
    ],
    [
        sg.Text("Name of connected device:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black"),
        sg.VSeparator(),
        sg.Text("Device state:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black")
    ],
    [sg.Text("-name-", key="devName", size=(30, 1)), sg.VSeparator(), sg.Text("-state-", key="devState", size=(30, 1))],
    [
        sg.Text("Version of newest bootloader:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black"),
        sg.VSeparator(),
        sg.Text("Flashed bootloader version:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black")
    ],
    [
        sg.Text("-version-", key="newBoot", size=(30, 1)),
        sg.VSeparator(),
        sg.Text("-version-", key="currBoot", size=(30, 1))
    ],
    [
        sg.Text("Current Depthai version:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black"),
        sg.VSeparator(),
        sg.Text("Current commit:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="black"),
    ],
    [
        sg.Text("-version-", key="version", size=(30, 1)),
        sg.VSeparator(),
        sg.Text("-version-", key="commit", size=(31, 1))
    ],
    [sg.HSeparator()],
    [
        sg.Text("", size=(7, 2)),
        sg.Button("Flash Newest Bootloader", size=(20, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500'),
    ]
]

# layout for config tab
deviceConfigLayout = [
    [sg.Text("Configuration settings", size=(20, 1), font=('Arial', 30, 'bold'), text_color="black")],
    [sg.HSeparator()],
    [
        sg.Button("About device", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_aboutBtn"),
        sg.Button("Config", size=(15, 1), font=('Arial', 10, 'bold'), disabled=True, key="_unique_configBtn"),
        sg.Button("Application", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False,  key="_unique_appBtn"),
        sg.Button("Danger Zone", size=(15, 1), font=('Arial', 10, 'bold'), button_color='#FF0500', disabled=False,  key="_unique_dangerBtn"),

    ],
    [sg.HSeparator()],
    [
        sg.Text("IPv4 type:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="ipTypeText"),
        sg.Radio('Static', "ipType", default=True, font=('Arial', 10, 'bold'), text_color="black",
                 key="staticBut", enable_events=True, disabled=True),
        sg.Radio('Dynamic', "ipType", default=False, font=('Arial', 10, 'bold'), text_color="black",
                 key="dynamicBut", enable_events=True, disabled=True)
    ],
    [
        sg.Text("IPv4:", size=(12, 1), font=('Arial', 10, 'bold'), text_color="gray", key="ipText"),
        sg.InputText(key="ip", size=(16, 2), disabled=True),
        sg.Text("Mask:", size=(5, 1), font=('Arial', 10, 'bold'), text_color="gray", key="maskText"),
        sg.InputText(key="mask", size=(16, 2), disabled=True),
        sg.Text("Gateway:", size=(8, 1), font=('Arial', 10, 'bold'), text_color="gray", key="gatewayText"),
        sg.InputText(key="gateway", size=(16, 2), disabled=True)
    ],
    [
        sg.Text("DNS name:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="dnsText"),
        sg.InputText(key="dns", size=(30, 2), disabled=True)
    ],
    [
        sg.Text("Alt DNS name:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="dnsAltText"),
        sg.InputText(key="dnsAlt", size=(30, 2), disabled=True)
    ],
    [
        sg.Text("Network timeout [ms]:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="networkTimeoutText"),
        sg.InputText(key="networkTimeout", size=(30, 2), disabled=True)
    ],
    [
        sg.Text("MAC address:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="macText"),
        sg.InputText(key="mac", size=(30, 2), disabled=True)
    ],
    [sg.HSeparator()],
    [
        sg.Text("USB timeout [ms]:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="usbTimeoutText"),
        sg.InputText(key="usbTimeout", size=(30, 2), disabled=True)
    ],
    [
        sg.Text("USB max speed:", size=(30, 1), font=('Arial', 10, 'bold'), text_color="gray", key="usbSpeedText"),
        sg.Combo(USB_SPEEDS, "Select speed", key="usbSpeed", size=(30, 6), disabled=True)
    ],
    [sg.HSeparator()],
    [
        sg.Text("", size=(8, 2)),
        sg.Button("Flash configuration", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500'),
        sg.Button("Clear configuration", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500'),
        sg.Button("View configuration", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500'),
    ],
]

# layout for app tab
appLayout = [
    [sg.Text("Application settings", size=(20, 1), font=('Arial', 30, 'bold'), text_color="black")],
    [sg.HSeparator()],
    [
        sg.Button("About device", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_aboutBtn"),
        sg.Button("Config", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_configBtn"),
        sg.Button("Application", size=(15, 1), font=('Arial', 10, 'bold'), disabled=True,  key="_unique_appBtn"),
        sg.Button("Danger Zone", size=(15, 1), font=('Arial', 10, 'bold'), button_color='#FF0500', disabled=False,  key="_unique_dangerBtn"),
    ],
    [sg.HSeparator()],
    [
        sg.Button("Flash application", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500'),
        sg.Button("Remove application", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                button_color='#FFA500'),
    ],
    [sg.HSeparator()],
    [
        sg.Button("Open device streaming application", size=(15, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500', key="startStreamingApp"),
    ],
]


# layout for app tab
dangerLayout = [
    [sg.Text("Danger Zone", size=(20, 1), font=('Arial', 30, 'bold'), text_color="black")],
    [sg.HSeparator()],
    [
        sg.Button("About device", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_aboutBtn"),
        sg.Button("Config", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False, key="_unique_configBtn"),
        sg.Button("Application", size=(15, 1), font=('Arial', 10, 'bold'), disabled=False,  key="_unique_appBtn"),
        sg.Button("Danger Zone", size=(15, 1), font=('Arial', 10, 'bold'), button_color='#FF0500', disabled=True,  key="_unique_dangerBtn"),
    ],
    [sg.HSeparator()],
    [
        sg.Button("Flash Factory Bootloader", size=(20, 2), font=('Arial', 10, 'bold'), disabled=True,
                  button_color='#FFA500', key='flashFactoryBootloader'),
        sg.Button("Factory reset",  size=(17, 2), font=('Arial', 10, 'bold'), disabled=True, button_color='#FFA500'),
        sg.Button("Boot into USB\nRecovery mode", size=(20, 2), font=('Arial', 10, 'bold'), disabled=True,
                  key='recoveryMode', button_color='#FFA500')
    ],
]


# layout of whole GUI with closed tabs set to false
layout = [
    [
        sg.Column(aboutDeviceLayout, key='-COL1-'),
        sg.Column(deviceConfigLayout, visible=False, key='-COL2-'),
        sg.Column(appLayout, visible=False, key='-COL3-'),
        sg.Column(dangerLayout, visible=False, key='-COL4-'),
    ]
]


class DeviceManager:
    devices: Dict[str, dai.DeviceInfo] = dict()
    values: Dict = None

    def __init__(self) -> None:
        self.window = sg.Window(title="Device Manager",
            icon="assets/icon.png",
            layout=layout,
            size=(645, 380),
            finalize=True # So we can do First search for devices
            )

        self.bl: dai.DeviceBootloader = None

        # First device search
        self.getDevices()

    def isPoE(self) -> bool:
        try:
            return self.bl.getType() == dai.DeviceBootloader.Type.NETWORK
        except Exception as ex:
            PrintException()
            sg.Popup(f'{ex}')

    def isUsb(self) -> bool:
        return not self.isPoE()

    def run(self) -> None:
        while True:
            event, self.values = self.window.read()
            if event == sg.WIN_CLOSED:
                break
            dev = self.values['devices']

            if event == "devices":
                if dev != "Select device":
                    # "allow flashing bootloader" boots latest bootloader first
                    # which makes the information of current bootloader, etc.. not correct (can be checked by "isEmbeddedVersion")
                    # So leave it to false, uncomment the isEmbeddedVersion below and only boot into latest bootlaoder upon the request to flash new bootloader
                    # bl = dai.DeviceBootloader(devices[values['devices']], False)
                    device = self.device
                    if deviceStateTxt(device.state) == "BOOTED":
                        # device is already booted somewhere else
                        sg.Popup("Device is already booted somewhere else!")
                    else:
                        self.resetGui()
                        self.bl = connectToDevice(device)
                        if self.bl is None: continue
                        self.getConfigs()
                        self.unlockConfig()
                else:
                    self.window.Element('progress').update("No device selected.")
            elif event == "Search":
                self.getDevices() # Re-search devices for dropdown
                selDev = SearchDevice()
                di = selDev.wait()
                if di is not None:
                    self.resetGui()
                    self.addDeviceInfo(di)  # Add device info to devices, if it's not there yet
                    self.window.Element('devices').update(di.getMxId())
                    _, self.values = self.window.read(1)
                    self.bl = connectToDevice(di)
                    print('after connecting to device,', self.bl)
                    if self.bl is None: continue
                    self.getConfigs()
                self.unlockConfig()
            elif event == "Specify IP":
                select = SelectIP()
                ok, ip = select.wait()
                if ok:
                    self.resetGui()
                    di = dai.DeviceInfo(ip)
                    di.state = dai.XLinkDeviceState.X_LINK_BOOTLOADER
                    di.protocol = dai.XLinkProtocol.X_LINK_TCP_IP
                    self.devices[ip] = di # Add to devices dict
                    self.window.Element('devices').update(ip) # Show to user
                    _, self.values = self.window.read(1)
                    self.bl = connectToDevice(di)
                    if self.bl is None: continue
                    self.getConfigs()
                self.unlockConfig()
            elif event == "Flash Newest Bootloader":
                # Use current type
                if flashBootloader(self.bl, self.device, self.bl.getType()):
                    # Device will reboot, close previous and reset GUI
                    self.closeDevice()
                    self.resetGui()
                    self.getDevices()
                else:
                    print("Flashing bootloader canceled.")

            # Danger
            elif event == "flashFactoryBootloader":
                sel = SelectBootloader(['AUTO', 'USB', 'NETWORK'], "Select bootloader type to flash.")
                ok, type = sel.wait()
                if ok:
                    # We will reconnect, as we need to set allowFlashingBootloader to True
                    self.closeDevice()
                    flashFactoryBootloader(self.device, type)
                    # Device will reboot, close previous and reset GUI
                    self.closeDevice()
                    self.resetGui()
                    self.getDevices()
                else:
                    print("Flashing Factory bootloader cancelled.")

            elif event == "Factory reset":
                sel = SelectBootloader(['NETWORK', 'USB'], "Select bootloader type used for factory reset.")
                ok, type = sel.wait()
                if ok:
                    # We will reconnect, as we need to set allowFlashingBootloader to True
                    self.closeDevice()
                    factoryReset(self.device, type)
                    # Device will reboot, close previous and reset GUI
                    self.closeDevice()
                    self.resetGui()
                    self.getDevices()
                else:
                    print("Factory reset cancelled.")

            elif event == "Flash configuration":
                self.flashConfig()
                self.getConfigs()
                self.resetGui()
                if self.isUsb():
                    self.unlockConfig()
                else:
                    self.devices.clear()
                    self.window.Element('devices').update("Search for devices", values=[])
            elif event == "Clear configuration":
                self.clearConfig()
                self.getConfigs()
                self.resetGui()
                if self.isUsb():
                    self.unlockConfig()
                else:
                    self.devices.clear()
                    self.window.Element('devices').update("Search for devices", values=[])
            elif event == "View configuration":
                try:
                    confJson = self.bl.readConfigData()
                    sg.popup_scrolled(confJson, title='Configuration')
                except Exception as ex:
                    sg.popup(f'No existing config to view ({ex})')

            elif event == "Flash application":
                file = sg.popup_get_file("Select .dap file", file_types=(('DepthAI Application Package', '*.dap'), ('All Files', '*.* *')))
                flashFromFile(file, self.bl)
            elif event == "Remove application":
                try:
                    self.bl.flashClear()
                    sg.popup(f'Successfully removed application')
                except Exception as ex:
                    sg.popup(f"Couldn't remove application ({ex})")

            elif event.startswith("_unique_configBtn"):
                self.window['-COL1-'].update(visible=False)
                self.window['-COL2-'].update(visible=True)
                self.window['-COL3-'].update(visible=False)
                self.window['-COL4-'].update(visible=False)
            elif event.startswith("_unique_aboutBtn"):
                self.window['-COL1-'].update(visible=True)
                self.window['-COL2-'].update(visible=False)
                self.window['-COL3-'].update(visible=False)
                self.window['-COL4-'].update(visible=False)
            elif event.startswith("_unique_appBtn"):
                self.window['-COL1-'].update(visible=False)
                self.window['-COL2-'].update(visible=False)
                self.window['-COL3-'].update(visible=True)
                self.window['-COL4-'].update(visible=False)
            elif event.startswith("_unique_dangerBtn"):
                self.window['-COL1-'].update(visible=False)
                self.window['-COL2-'].update(visible=False)
                self.window['-COL3-'].update(visible=False)
                self.window['-COL4-'].update(visible=True)


            elif event == "recoveryMode":
                if recoveryMode(self.bl):
                    sg.Popup(f'Device successfully put into USB recovery mode.')
                # Device will reboot, close previous and reset GUI
                self.closeDevice()
                self.resetGui()
                self.getDevices()
            elif event == "dynamicBut":
                # Clear out IPv4 settings to default to
                self.window.Element('ip').update('')
                self.window.Element('mask').update('')
                self.window.Element('gateway').update('')

            elif event == "startStreamingApp":
                # We will reconnect, as we need to set allowFlashingBootloader to True
                self.closeDevice()
                connectAndStartStreaming(self.device)
                self.resetGui()
                self.getDevices()

        self.window.close()

    @property
    def device(self) -> dai.DeviceInfo:
        """
        Get selected device
        """
        return self.devices[self.values['devices']]

    def addDeviceInfo(self, deviceInfo: dai.DeviceInfo):
        if deviceInfo.getMxId() not in self.devices:
            # Add the new deviceInfo
            self.devices[deviceInfo.getMxId()] = deviceInfo

    def getConfigs(self):
        device = self.device
        try:
            conf = self.bl.readConfig()
        except:
            conf = dai.DeviceBootloader.Config()

        try:
            if self.isPoE():
                if conf.isStaticIPV4():
                    self.window.Element('staticBut').update(True)
                else:
                    self.window.Element('dynamicBut').update(True)

                if conf.getIPv4() == '0.0.0.0':
                    self.window.Element('ip').update('')
                else:
                    self.window.Element('ip').update(conf.getIPv4())

                if conf.getIPv4Mask() == '0.0.0.0':
                    self.window.Element('mask').update('')
                else:
                    self.window.Element('mask').update(conf.getIPv4Mask())

                if conf.getIPv4Gateway() == '0.0.0.0':
                    self.window.Element('gateway').update('')
                else:
                    self.window.Element('gateway').update(conf.getIPv4Gateway())

                if conf.getDnsIPv4() == '0.0.0.0':
                    self.window.Element('dns').update('')
                else:
                    self.window.Element('dns').update(conf.getDnsIPv4())
                if conf.getDnsAltIPv4() == '0.0.0.0':
                    self.window.Element('dnsAlt').update('')
                else:
                    self.window.Element('dnsAlt').update(conf.getDnsAltIPv4())
                self.window.Element('networkTimeout').update(int(conf.getNetworkTimeout().total_seconds() * 1000))
                if conf.getMacAddress() == '00:00:00:00:00:00':
                    self.window.Element('mac').update('')
                else:
                    self.window.Element('mac').update(conf.getMacAddress())
                for el in CONF_INPUT_USB:
                    self.window[el].update("")
            else:
                for el in CONF_INPUT_POE:
                    self.window[el].update("")
                self.window.Element('usbTimeout').update(int(conf.getUsbTimeout().total_seconds() * 1000))
                self.window.Element('usbSpeed').update(str(conf.getUsbMaxSpeed()).split('.')[1])

            self.window.Element('devName').update(device.name)
            # self.window.Element('devNameConf').update(device.getMxId())
            self.window.Element('newBoot').update(dai.DeviceBootloader.getEmbeddedBootloaderVersion())

            # The "isEmbeddedVersion" tells you whether BL had to be booted,
            # or we connected to an already flashed Bootloader.
            if self.bl.isEmbeddedVersion():
                self.window.Element('currBoot').update('Not Flashed')
            else:
                self.window.Element('currBoot').update(self.bl.getVersion())

            self.window.Element('version').update(dai.__version__)
            self.window.Element('commit').update(dai.__commit__)
            self.window.Element('devState').update(deviceStateTxt(self.device.state))
        except Exception as ex:
            PrintException()
            sg.Popup(f'{ex}')

    def unlockConfig(self):
        if self.bl is None: return

        if self.isPoE():
            for el in CONF_INPUT_POE:
                self.window[el].update(disabled=False)
            for el in CONF_TEXT_POE:
                self.window[el].update(text_color="black")
        else:
            for el in CONF_INPUT_USB:
                self.window[el].update(disabled=False)
            for el in CONF_TEXT_USB:
                self.window[el].update(text_color="black")

        self.window['Flash Newest Bootloader'].update(disabled=False)
        self.window['flashFactoryBootloader'].update(disabled=False)
        self.window['Flash configuration'].update(disabled=False)
        self.window['Clear configuration'].update(disabled=False)
        self.window['View configuration'].update(disabled=False)
        self.window['Factory reset'].update(disabled=False)

        self.window['Flash application'].update(disabled=False)
        self.window['Remove application'].update(disabled=False)
        self.window['startStreamingApp'].update(disabled=False)

        self.window['recoveryMode'].update(disabled=False)

    def resetGui(self):
        """
        Reset GUI to its original state
        """
        for conf in [CONF_INPUT_POE, CONF_INPUT_USB]:
            for el in conf:
                self.window[el].update(disabled=True)
                self.window[el].update("")
        for conf in [CONF_TEXT_POE, CONF_TEXT_USB]:
            for el in conf:
                self.window[el].update(text_color="gray")

        self.window['Flash Newest Bootloader'].update(disabled=True)
        self.window['flashFactoryBootloader'].update(disabled=True)
        self.window['Flash configuration'].update(disabled=True)
        self.window['Clear configuration'].update(disabled=True)
        self.window['View configuration'].update(disabled=True)
        self.window['Factory reset'].update(disabled=True)
        self.window['Flash application'].update(disabled=True)
        self.window['Remove application'].update(disabled=True)
        self.window['startStreamingApp'].update(disabled=True)

        self.window['recoveryMode'].update(disabled=True)

        self.window.Element('devName').update("-name-")
        # self.window.Element('devNameConf').update("")
        self.window.Element('newBoot').update("-version-")
        self.window.Element('currBoot').update("-version-")
        self.window.Element('version').update("-version-")
        self.window.Element('commit').update("-version-")
        self.window.Element('devState').update("-state-")

    def closeDevice(self):
        if self.bl is not None:
            self.bl.close()
        self.bl = None

    def getDevices(self):
        self.closeDevice()  # If we are searching for new devices, first disconnect from the current one
        try:
            listedDevices = []
            self.devices.clear()
            deviceInfos = dai.XLinkConnection.getAllConnectedDevices()
            if not deviceInfos:
                self.window.Element('devices').update("No devices")
                sg.Popup("No devices found.")
            else:
                for deviceInfo in deviceInfos:
                    deviceTxt = deviceInfo.getMxId()
                    listedDevices.append(deviceTxt)
                    self.devices[deviceTxt] = deviceInfo

            # Update the list regardless
            self.window.Element('devices').update("Select device", values=listedDevices)
        except Exception as ex:
            PrintException()
            sg.Popup(f'{ex}')

    def flashConfig(self):
        values = self.values

        # Read modify write instead of flashing over
        # clearConfig can be used to start from scratch
        conf = dai.DeviceBootloader.Config()
        try:
            conf = self.bl.readConfig()
        except:
            pass

        try:
            if self.isPoE:
                if self.values['staticBut']:
                    if check_ip(values['ip']) and check_ip(values['mask']) and check_ip(values['gateway'], req=False):
                        conf.setStaticIPv4(values['ip'], values['mask'], values['gateway'])
                    else:
                        raise Exception('IP or Mask missing using static IP configuration')
                else:
                    if check_ip(values['ip'], req=False) and check_ip(values['mask'], req=False) and check_ip(values['gateway'], req=False):
                        conf.setDynamicIPv4(values['ip'], values['mask'], values['gateway'])

                conf.setDnsIPv4(values['dns'], values['dnsAlt'])
                if values['networkTimeout'] != "":
                    if int(values['networkTimeout']) >= 0:
                        conf.setNetworkTimeout(timedelta(seconds=int(values['networkTimeout']) / 1000))
                    else:
                        sg.Popup("Values can not be negative!")
                if values['mac'] != "":
                    if check_mac(values['mac']):
                        conf.setMacAddress(values['mac'])
                else:
                    conf.setMacAddress('00:00:00:00:00:00')
            else:
                if values['usbTimeout'] != "":
                    if int(values['usbTimeout']) >= 0:
                        conf.setUsbTimeout(timedelta(seconds=int(values['usbTimeout']) / 1000))
                    else:
                        sg.Popup("Values can not be negative!")
                if values['usbSpeed'] != "":
                    conf.setUsbMaxSpeed(getattr(dai.UsbSpeed, values['usbSpeed']))

            success, error = self.bl.flashConfig(conf)
            if not success:
                sg.Popup(f"Flashing failed: {error}")
            else:
                sg.Popup("Flashing successful.")
        except Exception as ex:
            PrintException()
            sg.Popup(f'{ex}')

    def clearConfig(self):
        try:
            success, error = self.bl.flashConfigClear()
            if not success:
                sg.Popup(f"Clearing configuration failed: {error}")
            else:
                sg.Popup("Successfully cleared configuration.")
        except Exception as ex:
            PrintException()
            sg.Popup(f'{ex}')


app = DeviceManager()
app.run()
