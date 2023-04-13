from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-si', action="store_true", help="Static input.")
parser.add_argument('-vid', action="store_true", help="Use video files")
parser.add_argument('-calib', type=str, default=None, help="Path to calibration file in json")
parser.add_argument('-left', type=str, default="left.png", help="left static input image")
parser.add_argument('-right', type=str, default="right.png", help="right static input image")
parser.add_argument('-bottom', type=str, default="bottom.png", help="bottom static input image")
parser.add_argument('-debug', action="store_true", default=False, help="Debug code.")
parser.add_argument('-rect', '--rectified', action="store_true", default=False, help="Generate and display rectified streams.")
parser.add_argument('-fps', type=int, default=10, help="Set camera FPS.")
parser.add_argument('-outDir', type=str, default="out", help="Output directory for depth maps and rectified files.")
parser.add_argument('-saveFiles', action="store_true", default=False, help="Save output files.")
parser.add_argument('-fullResolution', action="store_true", default=False, help="Use full resolution for depth maps.")
parser.add_argument('-xStart', type=int, default=None, help="X start coordinate for depth map.")
parser.add_argument('-yStart', type=int, default=None, help="Y start coordinate for depth map.")
parser.add_argument('-cs', type=int, default=128, help="Crop start.")
parser.add_argument('-imageCrop', default=None, choices=['center', 'right', 'left', 'top', 'bottom'], help="Select default crop for a part of the image")
parser.add_argument('-numLastFrames', type=int, default=None, help="Number of frames (last frames are used) for calculating the average depth.")

args = parser.parse_args()

forceFisheye = False

imageWidth = 1920
imageHeight = 1200

cropWidth = 1280
cropHeight = 800

cropLength = 1024
cropstart = args.cs
if cropstart + cropLength > cropWidth:
    cropstart = cropWidth - cropLength


staticInput = args.si

enableRectified = args.rectified
cameraFPS = args.fps
blockingOutputs = False

if args.imageCrop == "left":
    cropstart = 0
elif args.imageCrop == "right":
    cropstart = cropWidth - cropLength
print(f"Setting crop start to {cropstart}")

if args.saveFiles:
    Path(args.outDir).mkdir(parents=True, exist_ok=True)

if staticInput and args.vid:
    print("Static input and video input cannot be used at the same time.")
    exit(1)

if (args.fullResolution and not staticInput) and (args.fullResolution and not args.vid):
    print("Full resolution can only be used with static input or video input.")
    exit(1)

# if args.imageCrop and not args.fullResolution:
#     print("Image crop can only be used with full resolution.")
#     exit(1)


if args.fullResolution:
    if args.xStart is not None and args.yStart is not None and args.imageCrop:
        print("xStart and yStart will override imageCrop setting")
    if args.xStart is not None and args.yStart is not None:
        cropXStart = args.xStart
        cropYStart = args.yStart
    elif args.imageCrop == "left":
        cropXStart = 0
        cropYStart = (imageHeight - cropHeight) // 2
        cropstart = 0
    elif args.imageCrop == "right":
        cropXStart = (imageWidth - cropWidth)
        cropYStart = (imageHeight - cropHeight) // 2
        cropstart = cropWidth - cropLength
    elif args.imageCrop == "center":
        cropXStart = (imageWidth - cropWidth) // 2
        cropYStart = (imageHeight - cropHeight) // 2
        cropstart = (cropWidth - cropLength) // 2
    elif args.imageCrop == "top":
        cropXStart = (imageWidth - cropWidth) // 2
        cropYStart = 0
        cropstart = (cropWidth - cropLength) // 2
    elif args.imageCrop == "bottom":
        cropXStart = (imageWidth - cropWidth) // 2
        cropYStart = (imageHeight - cropHeight)
        cropstart = (cropWidth - cropLength) // 2

    cropXEnd = cropWidth + cropXStart
    cropYEnd = cropHeight + cropYStart
    print(f"cropXStart {cropXStart} - cropXEnd {cropXEnd}")
    print(f"cropYStart {cropYStart} - cropYEnd {cropYEnd}")

