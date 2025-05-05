import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "This DepthAI example uses stereo cameras for face detection with YuNet, calculating 3D spatial coordinates via triangulation. It visualizes bounding boxes, keypoints, and disparity in real-time."

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    args = parser.parse_args()

    return parser, args
