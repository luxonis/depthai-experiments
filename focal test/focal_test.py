#!/usr/bin/env python3

"""
Use 'T' to trigger autofocus, 'IOKL,.'
for manual exposure/focus:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase
"""

import os
#os.environ["DEPTHAI_LEVEL"] = "debug"

import cv2
import numpy as np
import argparse
import depthai as dai
import collections
import time
from itertools import cycle
from matplotlib import pyplot as plt, cm
from matplotlib import colors
from scipy import optimize
import matplotlib.animation as animation

def socket_type_pair(arg):
    socket, type = arg.split(',')
    if not (socket in ['rgb', 'left', 'right', 'camd', 'came', 'camf']):  raise ValueError("")
    if not (type in ['m', 'mono', 'c', 'color']): raise ValueError("")
    is_color = True if type in ['c', 'color'] else False
    return [socket, is_color]

parser = argparse.ArgumentParser()
parser.add_argument('-cams', '--cameras', type=socket_type_pair, nargs='+',
                    default=[['rgb', True], ['left', False], ['right', False], ['camd', True]],
                    help="Which camera sockets to enable, and type: c[olor] / m[ono]. "
                    "E.g: -cams rgb,m right,c . Default: rgb,c left,m right,m camd,c")
parser.add_argument('-mres', '--mono-resolution', type=int, default=800, choices={480, 400, 720, 800},
                    help="Select mono camera resolution (height). Default: %(default)s")
parser.add_argument('-cres', '--color-resolution', default='1080', choices={'720', '800', '1080', '1200', '1500', '1520', '1560', '4000', '4k', '5mp', '12mp', '48mp'},
                    help="Select color camera resolution / height. Default: %(default)s")
parser.add_argument('-rot', '--rotate', const='all', choices={'all', 'rgb', 'mono'}, nargs="?",
                    help="Which cameras to rotate 180 degrees. All if not filtered")
parser.add_argument('-fps', '--fps', type=float, default=30,
                    help="FPS to set for all cameras")
parser.add_argument('-ds', '--isp-downscale', default=1, type=int,
                    help="Downscale the ISP output by this factor")
parser.add_argument('-rs', '--resizable-windows', action='store_true',
                    help="Make OpenCV windows resizable. Note: may introduce some artifacts")
parser.add_argument('-raw', '--enable-raw', default=False, action="store_true",
                    help='Enable the RAW camera streams')
parser.add_argument('-lens', '--lens_pos', default=115, type=float,
                    help='Set the lens position for focal test')
parser.add_argument('-dif', '--diff', default=15, type=int,
                    help='Set the value for looping through lens of camera')
parser.add_argument('-hs', '--H_SIZE', default=2, type=int,
                    help='Set the horizontal crop')
parser.add_argument('-ws', '--W_SIZE', default=2, type=int,
                    help='Set the vertical crop')
parser.add_argument('-cd', '--lensWait', default=0.5, type=float,
                    help='Set the vertical crop')
parser.add_argument('-mts', '--showcropMatrix', default=False, action="store_true",
                    help='Display the animation of all cropped images.')
parser.add_argument('-dbd', '--detectBoard', default=False, action="store_true",
                    help='Detect board, not crop the image.')
parser.add_argument('-anim', '--displayAnimation', default=False, action="store_true",
                    help='Display the animation of lends position.')
parser.add_argument('-bord', '--id_board', default=[[[0],[1],[2],[3]], [[4],[5],[6],[7]], [[8],[9],[10],[11]], [[12],[13],[14],[15]]],
                    help='Give the list od ID of charucos, which you wanna crop in boards.')
args = parser.parse_args()

cam_list = []
cam_type_color = {}
print("Enabled cameras:")
for socket, is_color in args.cameras:
    cam_list.append(socket)
    cam_type_color[socket] = is_color
    print(socket.rjust(7), ':', 'color' if is_color else 'mono')

print("DepthAI version:", dai.__version__)
print("DepthAI path:", dai.__file__)

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.RGB,   # Or CAM_A
    'left' : dai.CameraBoardSocket.LEFT,  # Or CAM_B
    'right': dai.CameraBoardSocket.RIGHT, # Or CAM_C
    'camd' : dai.CameraBoardSocket.CAM_D,
    'came' : dai.CameraBoardSocket.CAM_E,
    'camf' : dai.CameraBoardSocket.CAM_F,
}

cam_socket_to_name = {
    'RGB'  : 'rgb',
    'LEFT' : 'left',
    'RIGHT': 'right',
    'CAM_D': 'camd',
    'CAM_E': 'came',
    'CAM_F': 'camf',
}

rotate = {
    'rgb'  : args.rotate in ['all', 'rgb'],
    'left' : args.rotate in ['all', 'mono'],
    'right': args.rotate in ['all', 'mono'],
    'camd' : args.rotate in ['all', 'rgb'],
    'came' : args.rotate in ['all', 'rgb'],
    'camf' : args.rotate in ['all', 'rgb'],
}

mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
    1200: dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    '720':  dai.ColorCameraProperties.SensorResolution.THE_720_P,
    '800':  dai.ColorCameraProperties.SensorResolution.THE_800_P,
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '1200': dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    '4000': dai.ColorCameraProperties.SensorResolution.THE_4000X3000,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '5mp': dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '48mp': dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}

def clamp(num, v0, v1):
    return max(v0, min(num, v1))


def get_sigma(self, frame_gray):
	# Check if the image is blurry
	dst_laplace = cv2.Laplacian(frame_gray, cv2.CV_64F)
	mu, sigma = cv2.meanStdDev(dst_laplace)
	return sigma * frame_gray.shape[0]


def gaussian(x, amplitude, mean, stddev, const):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2)) + const

def GetArUcoCorners(img: np.ndarray, enu_id: np.array):
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
	corners, ids, ret = cv2.aruco.detectMarkers(img, aruco_dict)
	#if len(ids) != 4:
		# print('Setup is not correct!')
		#return None
	try:
		center_dict = {}
		for e, id in enumerate(ids):
			for num in enu_id:
				if num==id:
					corners_block = corners[e][0]
					center = np.mean(corners_block, axis=0)
					center_dict[str(id[0])] = center
		if len(center_dict)!=4:
			return None
	except:
		return None
	return center_dict

def WarpImage(img: np.ndarray, pnts: dict, size: tuple, enu_id: np.array):
	src_pnts = np.array([
		pnts[str(enu_id[0][0])], pnts[str(enu_id[1][0])],
		pnts[str(enu_id[2][0])], pnts[str(enu_id[3][0])]
	], dtype=np.float32)
	dst_pnt = np.array([
		[0, 0], [size[0], 0],
		[0, size[1]], [size[0], size[1]]
	], dtype=np.float32)
	warp_mat = cv2.getPerspectiveTransform(src_pnts, dst_pnt)
	img = cv2.warpPerspective(img, warp_mat, size)
	return img
# Calculates FPS over a moving window, configurable
class FPS:
    def __init__(self, window_size=30):
        self.dq = collections.deque(maxlen=window_size)
        self.fps = 0

    def update(self, timestamp=None):
        if timestamp == None: timestamp = time.monotonic()
        count = len(self.dq)
        if count > 0: self.fps = count / (timestamp - self.dq[0])
        self.dq.append(timestamp)

    def get(self):
        return self.fps

# Start defining a pipeline
pipeline = dai.Pipeline()
# Uncomment to get better throughput
#pipeline.setXLinkChunkSize(0)

control = pipeline.createXLinkIn()
control.setStreamName('control')

cam = {}
xout = {}
xout_raw = {}
streams = []
for c in cam_list:
    xout[c] = pipeline.createXLinkOut()
    xout[c].setStreamName(c)
    streams.append(c)
    if cam_type_color[c]:
        cam[c] = pipeline.createColorCamera()
        cam[c].setResolution(color_res_opts[args.color_resolution])
        cam[c].setIspScale(1, args.isp_downscale)
        #cam[c].initialControl.setManualFocus(85) # TODO
        cam[c].isp.link(xout[c].input)
    else:
        cam[c] = pipeline.createMonoCamera()
        cam[c].setResolution(mono_res_opts[args.mono_resolution])
        cam[c].out.link(xout[c].input)
    cam[c].setBoardSocket(cam_socket_opts[c])
    # Num frames to capture on trigger, with first to be discarded (due to degraded quality)
    #cam[c].initialControl.setExternalTrigger(2, 1)
    #cam[c].initialControl.setStrobeExternal(48, 1)
    #cam[c].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)

    #cam[c].initialControl.setManualExposure(15000, 400) # exposure [us], iso
    # When set, takes effect after the first 2 frames
    #cam[c].initialControl.setManualWhiteBalance(4000)  # light temperature in K, 1000..12000
    # cam[c].initialControl.setMisc("stride-align", 1)
    # cam[c].initialControl.setMisc("scanline-align", 1)
    control.out.link(cam[c].inputControl)
    if rotate[c]:
        cam[c].setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    cam[c].setFps(args.fps)
    if args.enable_raw:
        raw_name = 'raw_' + c
        xout_raw[c] = pipeline.create(dai.node.XLinkOut)
        xout_raw[c].setStreamName(raw_name)
        if args.enable_raw:
            streams.append(raw_name)
        cam[c].raw.link(xout_raw[c].input)

if 0:
    print("=== Using custom camera tuning, and limiting RGB FPS to 10")
    pipeline.setCameraTuningBlobPath("/home/user/Downloads/tuning_color_low_light.bin")
    # TODO: change sensor driver to make FPS automatic (based on requested exposure time)
    cam['rgb'].setFps(10)

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    #print('Connected cameras:', [c.name for c in device.getConnectedCameras()])
    print('Connected cameras:', device.getConnectedCameraFeatures())
    cam_name = {}
    for p in device.getConnectedCameraFeatures():
        print(f' -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:', end='')
        print('auto ' if p.hasAutofocus else 'fixed', '- ', end='')
        print(*[type.name for type in p.supportedTypes])
        socket_name = cam_socket_to_name[p.socket.name]
        cam_name[socket_name] = p.sensorName
        if args.enable_raw:
            cam_name['raw_'+socket_name] = p.sensorName

    print('USB speed:', device.getUsbSpeed().name)

    print('IR drivers:', device.getIrDrivers())

    device.startPipeline(pipeline)

    q = {}
    fps_host = {}  # FPS computed based on the time we receive frames in app
    fps_capt = {}  # FPS computed based on capture timestamps from device
    for c in streams:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
        # The OpenCV window resize may produce some artifacts
        if args.resizable_windows:
            cv2.namedWindow(c, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(c, (640, 480))
        fps_host[c] = FPS()
        fps_capt[c] = FPS()

    controlQueue = device.getInputQueue('control')

    # Manual exposure/focus set step
    EXP_STEP = 500  # us
    ISO_STEP = 50
    LENS_STEP = 3
    DOT_STEP = 100
    FLOOD_STEP = 100
    DOT_MAX = 1200
    FLOOD_MAX = 1500

    # Defaults and limits for manual focus/exposure controls
    lensPos = args.lens_pos
    lensMin = 0
    lensMax = 255

    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600
    diff=args.diff
    W_SIZE  = args.W_SIZE
    # Number of pieces Vertically to each Horizontal  
    H_SIZE = args.H_SIZE

    dotIntensity = 0
    floodIntensity = 0

    awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
    anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
    effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'

    print("Cam:", *['     ' + c.ljust(8) for c in cam_list], "[host | capture timestamp]")

    capture_list = []
    while True:
        for c in streams:
            pkt = q[c].tryGet()
            if pkt is not None:
                fps_host[c].update()
                fps_capt[c].update(pkt.getTimestamp().total_seconds())
                width, height = pkt.getWidth(), pkt.getHeight()
                capture = c in capture_list
                if capture:
                    capture_file_info = ('capture_' + c + '_' + cam_name[c]
                         + '_' + str(width) + 'x' + str(height)
                         + '_exp_' + str(int(pkt.getExposureTime().total_seconds()*1e6))
                         + '_iso_' + str(pkt.getSensitivity())
                         + '_lens_' + str(pkt.getLensPosition())
                         + '_' + capture_time
                         + '_' + str(pkt.getSequenceNum())
                        )
                    capture_list.remove(c)
                    print()
                if c.startswith('raw_'):
                    # Resize is done to skip the +50 black lines we get with RVC3.
                    # TODO: handle RAW10/RAW12 to work with getFrame/getCvFrame
                    payload = pkt.getData().view(np.uint16).copy()
                    payload.resize(height, width)
                    if capture:
                        # TODO based on getType
                        filename = capture_file_info + '_10bit.bw'
                        print('Saving:', filename)
                        payload.tofile(filename)
                    # Full range for display, use bits [15:6] of the 16-bit pixels
                    frame = payload * (1 << 6)
                    # Debayer color for preview/png
                    if cam_type_color[c.split('_')[-1]]:
                        # See this for the ordering, at the end of page:
                        # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
                        # TODO add bayer order to ImgFrame getType()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
                    #frame = np.ascontiguousarray(bgr)  # just in case
                else:
                    if capture and args.enable_raw:
                        payload = pkt.getData()
                        filename = capture_file_info + '_NV12.yuv'
                        print('Saving:', filename)
                        payload.tofile(filename)
                    frame = pkt.getCvFrame()
                if capture:
                    filename = capture_file_info + '.png'
                    print('Saving:', filename)
                    cv2.imwrite(filename, frame)
                cv2.imshow(c, frame)
        print("\rFPS:",
              *["{:6.2f}|{:6.2f}".format(fps_host[c].get(), fps_capt[c].get()) for c in cam_list],
              end=' ', flush=True)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_list = streams.copy()
            capture_time = time.strftime('%Y%m%d_%H%M%S')
        elif key == ord('x'):
            ctrl = dai.CameraControl()
            stopped = not stopped
            print("Streaming", "stop" if stopped else "start")
            if stopped:
                ctrl.setStopStreaming()
            else:
                ctrl.setStartStreaming()
            controlQueue.send(ctrl)
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)
        elif key == ord('f'):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            controlQueue.send(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
        elif key == ord('m'):
            print("\nStart focus test")
            auto_cameras={}
            manual_cameras={}
            lens_position={}
            frame_matrix={}
            if args.showcropMatrix:
                crop_matrix={}
            if args.detectBoard:
                board_matrix={}
                lends_detected={}
            index=0
            print(cam_list)
            for p in device.getConnectedCameraFeatures():
                if p.hasAutofocus:
                    auto_cameras[cam_list[index]]={}
                else:
                    manual_cameras[cam_list[index]]={}
                index+=1
            if auto_cameras=={}:
                print("There are no autofocus cameras, test will end.")
            if manual_cameras=={}:
                print("There are no manual cameras, test will end.")
            for c in cam_list:
                if auto_cameras == {} and manual_cameras == {}:
                    break
                if auto_cameras.get(c) is not None:
                    pkt=None
                    while pkt==None:
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            if args.lens_pos==115 and pkt.getLensPosition()>diff:
                                lensPos=pkt.getLensPosition()
                            elif pkt.getLensPosition()<diff:
                                print(f"Tracking lens position for {c} is not working properly, setting default value {args.lens_pos}")
                            else:
                                lensPos=args.lens_pos
                            lens_position[c]=[lensPos,lensPos-diff,lensPos+diff]
                            print(f"Starting lens position {lensPos} for camera {c}")
                            if args.showcropMatrix:
                                crop_matrix[c]={}
                            if args.detectBoard:
                                board_matrix[c]={}
                                lends_detected[c]={}
                                frame_matrix[c]={}
                            else:
                                auto_cameras[c]={}
                                frame_matrix[c]=[]
                            break
                if manual_cameras.get(c) is not None:
                    while pkt==None:
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            manual_cameras[c]={}
            index=0
            id_boards=args.id_board
            def Sobel_filter(frame):
                sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
                ## Calculate gradient magnitude
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                ## Normalize the gradient magnitude
                gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                return gradient_magnitude

            while index<2*diff+1:
                if auto_cameras=={}:
                    break
                for c in cam_list:
                    if auto_cameras.get(c) is not None and index!=0:
                        lensPos = clamp(lens_position[c][0], lens_position[c][2], lens_position[c][1])
                        print(f"Setting manual focus for camera {c}, lens position: ", lens_position[c][1])
                        ctrl = dai.CameraControl()
                        ctrl.setManualFocus(int(lens_position[c][1]))
                        controlQueue.send(ctrl)
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            frame = pkt.getCvFrame()
                            if pkt.getType() != dai.RawImgFrame.Type.RAW8:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if args.detectBoard:
                                if auto_cameras[c] is None:
                                    auto_cameras[c]={}
                                frame_index=0
                                for enu_id in id_boards:
                                    if auto_cameras[c].get(frame_index) is None:
                                        auto_cameras[c][frame_index]=[]
                                        lends_detected[c][frame_index]=[]
                                        frame_matrix[c][frame_index]=[]
                                    try:
                                        cnrs = GetArUcoCorners(frame, enu_id)
                                        img_warp = WarpImage(frame, cnrs, (140, 70), enu_id)
                                        dst_laplace = cv2.Laplacian(img_warp, cv2.CV_64F).var()
                                        auto_cameras[c][frame_index].append(dst_laplace)
                                        lends_detected[c][frame_index].append(lens_position[c][1])
                                        frame_matrix[c][frame_index].append(Sobel_filter(img_warp))
                                    except:
                                        print(f"Board {enu_id} for {c} not found.")
                                    frame_index+=1
                            else:
                                frame_matrix[c].append(Sobel_filter(frame))
                                frame2 = frame
                                height, width = frame.shape[:2]
                                crop_index=0
                                for ih in range(H_SIZE):
                                   for iw in range(W_SIZE):
                                        if auto_cameras[c].get(crop_index) is None:
                                            auto_cameras[c][crop_index]=[]
                                        if args.showcropMatrix:
                                            if crop_matrix[c].get(crop_index) is None:
                                                crop_matrix[c][crop_index]=[]
                                        x = width / W_SIZE * iw 
                                        y = height / H_SIZE * ih
                                        h = (height / H_SIZE)
                                        w = (width / W_SIZE )
                                        frame = frame[int(y):int(y+h), int(x):int(x+w)]
                                        NAME = str(time.time()) 
                                        dst_laplace = cv2.Laplacian(frame, cv2.CV_64F).var()
                                        auto_cameras[c][crop_index].append(dst_laplace)
                                        if args.showcropMatrix:
                                            crop_matrix[c][crop_index].append(Sobel_filter(frame))
                                        frame = frame2
                                        crop_index+=1
                    if auto_cameras.get(c) is not None:
                        lens_position[c][1]=lens_position[c][1]+1
                    if manual_cameras.get(c) is not None and index!=0:
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            frame = pkt.getCvFrame()
                            if pkt.getType() != dai.RawImgFrame.Type.RAW8:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if args.detectBoard:
                                if manual_cameras[c] is None:
                                    manual_cameras[c]={}
                                frame_index=0
                                for enu_id in id_boards:
                                    if manual_cameras[c].get(frame_index) is None:
                                        manual_cameras[c][frame_index]=[]
                                    try:
                                        cnrs = GetArUcoCorners(frame, enu_id)
                                        img_warp = WarpImage(frame, cnrs, (140, 70), enu_id)
                                        dst_laplace = cv2.Laplacian(img_warp, cv2.CV_64F).var()
                                        manual_cameras[c][frame_index].append(dst_laplace)
                                    except:
                                        print(f"Board {enu_id} for {c} not found.")
                                    frame_index+=1
                            else:
                                frame2 = frame
                                height, width = frame.shape[:2]
                                crop_index=0
                                for ih in range(H_SIZE):
                                   for iw in range(W_SIZE):
                                        if manual_cameras[c].get(crop_index) is None:
                                            manual_cameras[c][crop_index]=[]
                                        x = width / W_SIZE * iw 
                                        y = height / H_SIZE * ih
                                        h = (height / H_SIZE)
                                        w = (width / W_SIZE )
                                        frame = frame[int(y):int(y+h), int(x):int(x+w)]
                                        NAME = str(time.time()) 
                                        dst_laplace = cv2.Laplacian(frame, cv2.CV_64F).var()
                                        manual_cameras[c][crop_index].append(dst_laplace)
                                        frame = frame2
                                        crop_index+=1
                time.sleep(args.lensWait)
                index+=1
            if auto_cameras!={}:
                camera_index=0
                for c, crop in auto_cameras.items():
                    fig, (ax1, ax2) = plt.subplots(2)
                    if args.showcropMatrix:
                        fig2 =plt.figure()
                    # Create an empty image object
                    if args.detectBoard:
                        for crop_index, variance  in crop.items():
                            #fig, (ax1, ax2) = plt.subplots(2)
                            lens=lends_detected[c][crop_index]
                            try:
                                popt, _ = optimize.curve_fit(gaussian, lens, variance, p0=(max(variance), (lens_position[c][1]+lens_position[c][2]-2*diff+2)/2, 5, min(variance)))
                                ax2.plot(lens,variance, "-x", color=plt.cm.nipy_spectral(crop_index/len(crop.items())), label=f"CAM: {c}, board {crop_index} Autofocus: {(lens_position[c][1]+lens_position[c][2]-2*diff)/2}, Estimated:  {round(popt[1],2)}")
                                ax2.plot(lens, gaussian(lens, *popt), color=plt.cm.nipy_spectral(crop_index/len(crop.items())))
                                if args.displayAnimation:
                                    img = ax1.imshow(frame_matrix[c][2], cmap='hot')
                                    cbar = fig.colorbar(img)
                                    title = ax1.set_title(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2}')
                                    def update_matrix(i):
                                        # Get the matrix for the current frame
                                        current_matrix = frame_matrix[c][i % len(frame_matrix[c])+2]

                                        # Update the image data with the current matrix
                                        img.set_array(current_matrix)
                                        title.set_text(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2+i}')

                                    # Create the animation
                                    ani1 = animation.FuncAnimation(fig, update_matrix, frames=len(frame_matrix[c])-2, interval=100)
                                else:
                                    title = ax1.set_title(f'Autofocus, lends position  {round(popt[1],2)}')
                                    ax1.imshow(frame_matrix[c][crop_index][variance.index(max(variance))], cmap="hot")
                                camera_index+=1
                            except:
                                print(f"Could not calculate the focus range with variance {variance} and lens position {lens}.")
                            #ax2.set_title("Focal test for camera")
                            #ax2.set_xlabel("Lens position")
                            #ax2.set_ylabel("Laplacian variance")
                            #ax2.grid()
                            #ax2.legend()
                    else:#
                        if args.displayAnimation:
                            img = ax1.imshow(frame_matrix[c][2], cmap='hot')
                            cbar = fig.colorbar(img)
                            title = ax1.set_title(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2}')
                            def update_matrix(i):
                                # Get the matrix for the current frame
                                current_matrix = frame_matrix[c][i % len(frame_matrix[c])+2]

                                # Update the image data with the current matrix
                                img.set_array(current_matrix)
                                title.set_text(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2+i}')

                            # Create the animation
                            ani1 = animation.FuncAnimation(fig, update_matrix, frames=len(frame_matrix[c])-2, interval=100)
                        for crop_index, variance  in crop.items():
                            variance=variance[2:]
                            lens=np.arange(lens_position[c][1]-2*diff+1,lens_position[c][2])
                            popt, _ = optimize.curve_fit(gaussian, lens, variance, p0=(max(variance), (lens_position[c][1]+lens_position[c][2]-2*diff+2)/2, 5, min(variance)))
                            ax2.plot(lens,variance, "-x", color=plt.cm.nipy_spectral(crop_index/len(crop.items())), label=f"CAM: {c}, crop {crop_index}, Autofocus: {(lens_position[c][1]+lens_position[c][2]-2*diff)/2}, Estimated:  {round(popt[1],2)}")
                            ax2.plot(lens, gaussian(lens, *popt), color=plt.cm.nipy_spectral(crop_index/len(crop.items())))
                            if args.showcropMatrix:
                                ax3 = fig2.add_subplot(H_SIZE,W_SIZE,crop_index+1)
                                if args.displayAnimation:
                                    img2 = ax3.imshow(crop_matrix[c][crop_index][2], cmap='hot')
                                    title2 = fig2.suptitle(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2}')
                                    def update_matrix(i):
                                        # Get the matrix for the current frame
                                        current_matrix = crop_matrix[c][crop_index][i % len(crop_matrix[c][crop_index])+2]
                                    # Update the image data with the current matrix
                                        img2.set_data(current_matrix)
                                        title2.set_text(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2+i}')
                                    # Create the animation
                                    ani2 = animation.FuncAnimation(fig, update_matrix, frames=len(crop_matrix[c][crop_index])-2, interval=100)
                                else:
                                    ax3.imshow(crop_matrix[c][crop_index][variance.index(max(variance))], cmap='hot')
                            else:
                                title = ax1.set_title(f'Autofocus, lends position  {round(popt[1],2)}')
                                ax1.imshow(frame_matrix[c][variance.index(max(variance))], cmap="hot")

                            # Display the animation
                    ax2.set_title("Focal test for camera")
                    ax2.set_xlabel("Lens position")
                    ax2.set_ylabel("Laplacian variance")
                    ax2.grid()
                    ax2.legend()
                #ani.save("D:\\FAKS, FMF\\Studentska dela\\TLI.gif", dpi=300, writer=animation.PillowWriter(fps=25))
                try:
                    if abs(round(popt[1],2)-(lens_position[c][1]+lens_position[c][2]-2*diff)/2)>4:
                        print("Camera's autofocus does not work properly. If Gaussian is not centered press \"f\" and wait a few seconds. \nIf Gaussian is centered and the values still does not match, contact Luxonis.")
                    else:
                        print("Camera has no lends tilt.")
                except:
                    print("Focus not tested properly")
            print("Before running the test again, press F to restart autofocus.")
            if manual_cameras!={}:
                compare_variance = {}
                for c, crop in manual_cameras.items():
                    fig_man, ax1_man = plt.subplots()
                    reference_value = 0
                    camera_variance = []
                    for crop_index, variance  in crop.items():
                        avg_variance=sum(variance)/len(variance)
                        fig_man.suptitle(f"Manual camera focus of image, camera {c}")
                        n, bins, patches = ax1_man.hist(variance, bins=5, alpha=0.5, ec="black", label=f"Crop num {crop_index}, average {avg_variance}")
                        if crop_index == 0:
                            reference_value = avg_variance
                        else:
                            if abs(reference_value-avg_variance) > 1000:
                                print(f"Camera {c} has some tilt, between {crop_index} and {0}")
                        camera_variance.append(avg_variance)
                    ax1_man.legend()
                    ax1_man.grid()
                    ax1_man.set_xlabel("Lap variance")
                    ax1_man.set_ylabel("Num picture")
                    compare_variance[c]=sum(camera_variance)/len(camera_variance)
                print(compare_variance)
            plt.show()
            camera_index = 0
            reference_value = 0
            reference_camera = 0
            for c, variance in compare_variance.items():
                if camera_index == 0:
                    reference_value = variance
                    refrence_camera = c
                else:
                    if abs(reference_value-variance)>1000:
                        print(f"Camera has some tilt: {reference_camera} and {c} variance does not match.")
                camera_index+=1
        elif key == ord('r'):
            print("\nStart focus test")
            auto_cameras={}
            lens_position={}
            frame_matrix={}
            crop_matrix={}
            final_position={}
            max_sigma={}
            focus_set={}
            index=0
            which_frame=5

            for p in device.getConnectedCameraFeatures():
                if p.hasAutofocus:
                    auto_cameras[cam_list[index]]={}
                index+=1
            if auto_cameras=={}:
                print("There are no autofocus cameras, test will end.")
            for c in auto_cameras.keys():
                if auto_cameras=={}:
                    break
                if auto_cameras.get(c) is not None:
                    pkt=None
                    while pkt==None:
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            if args.lens_pos==115 and pkt.getLensPosition()>diff:
                                lensPos=pkt.getLensPosition()
                            elif pkt.getLensPosition()<diff:
                                print(f"Tracking lens position for {c} is not working properly, setting default value {args.lens_pos}")
                            else:
                                lensPos=args.lens_pos
                            lens_position[c]=[lensPos,lensPos-diff,lensPos+diff]
                            print(f"Starting lens position {lensPos} for camera {c}")
                            auto_cameras[c]={}
                            frame_matrix[c]=[]
                            crop_matrix[c]={}
                            max_sigma[c]={}
                            focus_set[c]=0
                            final_position[c]=False
                            break
            index=0
            jump=0
            def Sobel_filter(frame):
                sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
                ## Calculate gradient magnitude
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                ## Normalize the gradient magnitude
                gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                return gradient_magnitude

            while final_position[c]==False:
                if auto_cameras=={}:
                    break
                for c in auto_cameras.keys():
                    if auto_cameras.get(c) is not None and index!=0:
                        lensPos = clamp(lens_position[c][0], lens_position[c][2], lens_position[c][1])
                        print(f"Setting manual focus for camera {c}, lens position: ", lens_position[c][1])
                        ctrl = dai.CameraControl()
                        ctrl.setManualFocus(int(lens_position[c][1]))
                        controlQueue.send(ctrl)
                        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
                        pkt = q[c].tryGet()
                        if pkt is not None:
                            frame = pkt.getCvFrame()
                            if pkt.getType() != dai.RawImgFrame.Type.RAW8:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_matrix[c].append(Sobel_filter(frame))
                            frame2 = frame
                            height, width = frame.shape[:2]
                            crop_index=0
                            for ih in range(H_SIZE):
                               for iw in range(W_SIZE):
                                    if auto_cameras[c].get(crop_index) is None:
                                        auto_cameras[c][crop_index]=[]
                                    if crop_matrix[c].get(crop_index) is None:
                                        crop_matrix[c][crop_index]=[]
                                    x = width / W_SIZE * iw 
                                    y = height / H_SIZE * ih
                                    h = (height / H_SIZE)
                                    w = (width / W_SIZE )
                                    frame = frame[int(y):int(y+h), int(x):int(x+w)]
                                    NAME = str(time.time()) 
                                    dst_laplace = cv2.Laplacian(frame, cv2.CV_64F).var()
                                    auto_cameras[c][crop_index].append(dst_laplace)
                                    crop_matrix[c][crop_index].append(Sobel_filter(frame))
                                    frame = frame2
                                    crop_index+=1
                                    #print(f"Value of laplacian is {dst_laplace}, gradient magnitude {gradient_magnitude.var()}")
                                    #cv2.imshow(f"Focal test {c}, {lensPos}", frame)
                            sigma=auto_cameras[c][which_frame][index]
                            print(f"Lends position {lens_position[c][1]}")
                            if auto_cameras[c][which_frame][index]<auto_cameras[c][which_frame][index-1] and index>0:
                                if index==1: #If we went the wrong way at the start TODO If the jump was just because of noise
                                    diff=-diff 
                                elif index>1: #If we already went over the peak
                                    diff=-1
                                jump+=1
                                if jump==2 and auto_cameras[c][which_frame][index]>max_sigma[c]: #If we went over the peak twice, it sets the value for that camera
                                    final_position[c]==True
                                    focus_set[c]=lens_position[c][1]
                        if max_sigma[c]<auto_cameras[c][which_frame][index]:
                            max_sigma[c]=auto_cameras[c][which_frame][index]
                        lens_position[c][1]+=diff #Change lends position 
                        index+=1
                            
                time.sleep(args.lensWait)
                index+=1
            for c in auto_cameras.keys():
                print(f"Position of lends: camera {c} lends {focus_set[c]}")
            if auto_cameras!={}:
                for c, crop in auto_cameras.items():
                    fig, (ax1, ax2) = plt.subplots(2)
                    fig2 =plt.figure()

                    # Create an empty image object
                    img = ax1.imshow(frame_matrix[c][2], cmap='hot')
                    cbar = fig.colorbar(img)
                    title = ax1.set_title(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2}')
                    def update_matrix(i):
                        # Get the matrix for the current frame
                        current_matrix = frame_matrix[c][i % len(frame_matrix[c])+2]

                        # Update the image data with the current matrix
                        img.set_array(current_matrix)
                        title.set_text(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2+i}')

                    # Create the animation
                    ani1 = animation.FuncAnimation(fig, update_matrix, frames=len(frame_matrix[c])-2, interval=100)
                    for crop_index, variance  in crop.items():
                        variance=variance[2:]
                        lens=np.arange(lens_position[c][1]-2*diff+1,lens_position[c][2])
                        popt, _ = optimize.curve_fit(gaussian, lens, variance, p0=(max(variance), (lens_position[c][1]+lens_position[c][2]-2*diff+2)/2, 5, min(variance)))
                        ax3 = fig2.add_subplot(H_SIZE,W_SIZE,crop_index+1)
                        img2 = ax3.imshow(crop_matrix[c][crop_index][2], cmap='hot')
                        title2 = fig2.suptitle(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2}')
                        def update_matrix(i):
                            # Get the matrix for the current frame
                            current_matrix = crop_matrix[c][crop_index][i % len(crop_matrix[c][crop_index])+2]

                            # Update the image data with the current matrix
                            img2.set_data(current_matrix)
                            title2.set_text(f'Autofocus, lends position  {lens_position[c][1]-2*diff+2+i}')

                        # Create the animation
                        ani2 = animation.FuncAnimation(fig, update_matrix, frames=len(crop_matrix[c][crop_index])-2, interval=100)
                        # Display the animation
                        ax2.plot(lens,variance, "-x", color=plt.cm.nipy_spectral(crop_index/len(crop.items())), label=f"CAM: {c}, crop {crop_index}, Autofocus: {(lens_position[c][1]+lens_position[c][2]-2*diff)/2}, Estimated:  {round(popt[1],2)}")
                        ax2.plot(lens, gaussian(lens, *popt), color=plt.cm.nipy_spectral(crop_index/len(crop.items())))
                    ax2.set_title("Focal test for camera")
                    ax2.set_xlabel("Lens position")
                    ax2.set_ylabel("Laplacian variance")
                    ax2.grid()
                    ax2.legend()
                plt.show()
                #ani.save("D:\\FAKS, FMF\\Studentska dela\\TLI.gif", dpi=300, writer=animation.PillowWriter(fps=25))
                if abs(round(popt[1],2)-(lens_position[c][1]+lens_position[c][2]-2*diff)/2)>LENS_STEP:
                    print("Camera's autofocus does not work properly. If Gaussian is not centered press \"f\" and wait a few seconds. \nIf Gaussian is centered and the values still does not match, contact Luxonis.")
            print("Before running the test again, press F to restart autofocus.")