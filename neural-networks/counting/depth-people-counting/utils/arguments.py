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
        "-r",
        "--recording",
        help="Path to the recording directory you aim to run the experiment on. Recording should contain left.mp4 and right.mp4 files and calib.json file.",
        required=False,
        default=None,
        type=str,
    )

    args = parser.parse_args()

    return parser, args
