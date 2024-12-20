import cv2
import argparse
import depthai as dai
from pathlib import Path
from host_manual_camera_control import ManualCameraControl

parser = argparse.ArgumentParser()
parser.add_argument('-res', '--resolution', default='1080', choices={'1080', '4k', '12mp', '13mp'},
                    help="Select RGB resolution. Default: %(default)s")
parser.add_argument('-raw', '--enable_raw', default=False, action="store_true",
                    help='Enable the color RAW stream')
parser.add_argument('-fps', '--fps', default=30, type=int,
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

rgb_res_opts = {
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k'  : dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
}

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    if args.camera_tuning:
        pipeline.setCameraTuningBlobPath(str(args.camera_tuning))

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(rgb_res_opts.get(args.resolution))
    focus_name = 'af'
    if args.lens_position >= 0:
        cam.initialControl.setManualFocus(args.lens_position)
        focus_name = 'f' + str(args.lens_position)
    cam.setIspScale(1, args.isp_downscale)
    cam.setFps(args.fps)
    if args.rotate:
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

    cv2.namedWindow("isp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("isp", (960, 540))
    if args.enable_raw:
        cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("raw", (960, 540))

    pipeline.create(ManualCameraControl).build(
        preview_isp=cam.isp,
        preview_raw=cam.raw,
        control_queue=cam.inputControl.createInputQueue(),
        fps=args.fps,
        enable_raw=args.enable_raw,
        focus_name=focus_name
    )

    print("Pipeline created.")
    pipeline.run()