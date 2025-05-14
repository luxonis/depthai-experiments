import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps-limit",
        help="FPS limit.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--left",
        default=False,
        action="store_true",
        help="Enable left camera stream.",
    )
    parser.add_argument(
        "-r",
        "--right",
        default=False,
        action="store_true",
        help="Enable right camera stream.",
    )
    parser.add_argument(
        "-pc",
        "--pointcloud",
        default=False,
        action="store_true",
        help="Enable pointcloud stream.",
    )
    parser.add_argument(
        "-nc",
        "--no-color",
        default=False,
        action="store_true",
        help="Disable color camera stream.",
    )

    args = parser.parse_args()

    return parser, args
