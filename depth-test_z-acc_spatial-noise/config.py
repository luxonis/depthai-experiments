import depthai as dai
from argparse import ArgumentParser
from pathlib import Path
import ast
parser = ArgumentParser(prog = "Depth test", description = "Test test z-accuracy and spatial noise")

parser.add_argument("-c", "--color", action = "store_true", help = "Use color camera for preview")
parser.add_argument("-lr", "--lrcheck", action = "store_true", help = "Left-rigth check for better handling for occlusions")
parser.add_argument("-e", "--extended", action = "store_true", help = "Closer-in minimum depth, disparity range is doubled")
parser.add_argument("-s", "--subpixel", action = "store_true", help = "Better accuracy for longer distance, fractional disparity 32-levels")
parser.add_argument("-ct", "--confidence_threshold", type = int, default = 250, help = "0-255, 255 = low confidence, 0 = high confidence")
parser.add_argument("-mr", "--min_range", type = int, default = 100, help = "mm")
parser.add_argument("-xr", "--max_range", type = int, default = 2000, help = "mm")
parser.add_argument("-ms", "--mono_camera_resolution", type = str, default = "THE_400_P", choices=["THE_400_P", "THE_480_P", "THE_720_P", "THE_800_P"], help = "Mono camera resolution")
parser.add_argument("-m", "--median", type = str, default = "KERNEL_7x7", choices=["MEDIAN_OFF", "KERNEL_3x3", "KERNEL_5x5", "KERNEL_7x7"], help = "Median filter")
parser.add_argument("-n", "--n_samples", type = int, default = 10, help = "Number of samples in a single test")
parser.add_argument("-gt", "--ground_truth", type = float, default = 1, help = "Ground truth depth in m (camera wall distance)")
parser.add_argument("--astra_gt", action="store_true", default = None, help = "Use astra pro as ground truth")
parser.add_argument("--astra_intrinsic", type = str, default = None, help = "Path to astra intrinsic file (.np file containing 3x3 matrix)")
parser.add_argument('-calib', type=str, required=False, default=None, help="Path to calibration file in json")
parser.add_argument('-depth', type=str, required=False, default=None, help="Path to depth numpy file")
parser.add_argument('-rectified', type=str, required=False, default=None, help="Path to the rectified nu")
parser.add_argument('-out_results_f', type=str, required=False, default=None, help="Path to the output results file")
parser.add_argument('-roi_file', type=str, default=None, help = "Output ROI to a file")
parser.add_argument('-set_roi_file', type=str, default=None, help = "Set ROI from a file")
parser.add_argument('-set_roi_file_undistorted', type=str, default=None, help = "Set ROI from a file - undistorted")
parser.add_argument('-mode', type=str, default="interactive", choices=["interactive", "measure"], help = "Mode to run the test in" )
parser.add_argument('-vertical', action="store_true", help = "Run the test in vertical mode")
parser.add_argument('-use_opencv', action="store_true", help = "Use opencv for disparity")
parser.add_argument('-rs', type=float, default=1, help = "Resize windows by a factor")
parser.add_argument('-p', '--path', default = None, type = str, help = "Path to the recording folder")
parser.add_argument('-d', '--depth', default = 1.0, type = float, help = "Ground truth distance in meters")
parser.add_argument('-a', '--area', type = str, default = 'center', choices = ['center', 'right', 'left'], help = 'Choose the area you want to test')
parser.add_argument('-ocv', '--opencv', action = 'store_true', help = 'Use opencv for replay')
parser.add_argument('-calp', '--calibration_path', type = str, default = None, help = 'Path to the calibration file')
parser.add_argument('-alpha', type=float, default=0, help="alpha parameter for rectified frames")
parser.add_argument('-vert', "--vertical", action = "store_true", help = "Use vertical stereo mode")

args = parser.parse_args()

COLOR = args.color 				# Use color camera or mono camera for preview

# DEPTH CONFIG
mode = args.mode
roi_undistorted = False
if mode == "measure":
    if not args.set_roi_file and not args.set_roi_file_undistorted:
        print("One of the ROI files must be set")
        exit(1)
    # if not Path(args.set_roi_file).exists() or not Path(args.set_roi_file_undistorted).exists():
    #     print("ROI file not found: ", args.set_roi_file)
    #     exit(1)
    if args.set_roi_file and args.set_roi_file_undistorted:
        print("Only one ROI file can be set")
        exit(1)
    if args.set_roi_file:
        roi_file = args.set_roi_file
    elif args.set_roi_file_undistorted:
        roi_file = args.set_roi_file_undistorted
        roi_undistorted = True
    else:
        assert False

    with open(roi_file, 'r') as f:
        roi = ast.literal_eval(f.read().strip())

    if len(roi) != 4:
        print("ROI file should contain 4 values: ", args.set_roi_file)
        exit(1)

lrcheck  = args.lrcheck   			# Better handling for occlusions
extended = args.extended  			# Closer-in minimum depth, disparity range is doubled
subpixel = args.subpixel   			# Better accuracy for longer distance, fractional disparity 32-levels
confidence_threshold = args.confidence_threshold 	# 0-255, 255 = low confidence, 0 = high confidence
min_range = args.min_range 			# mm
max_range = args.max_range			# mm
mono_camera_resolution = getattr(dai.MonoCameraProperties.SensorResolution, args.mono_camera_resolution)


use_opencv = args.opencv
use_vertical = args.vertical
calibration_path = args.calibration_path
# Median filter
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = getattr(dai.StereoDepthProperties.MedianFilter, args.median)


n_samples = args.n_samples
alpha = args.alpha

gt = args.ground_truth

resuls_file = Path(__file__).parent / "results" / "results.txt"
if args.out_results_f:
    resuls_file = Path(args.out_results_f)
resuls_file.parent.mkdir(parents=True, exist_ok=True)

# Astra
astra_gt = args.astra_gt # use astra pro as ground truth
astra_intrinsic = args.astra_intrinsic # path to the astra intrinsic matrix
# path = args.path # path to the recording folder
area = args.area
path = args.path # path to the recording folder
real_depth = args.depth


# left (id 2. aka camc) - right (id 1 a.k.a camb)
#         |
#       below (id 0 a.k.a cama)
