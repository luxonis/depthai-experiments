import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = "This experiment showcases one possible approach for measuring the size of a box using DepthAI."

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-maxd",
        "--max-dist",
        type=float,
        default=2,
        help="maximum distance between camera and object in space in meters",
    )

    parser.add_argument(
        "-mins",
        "--min-box-size",
        type=float,
        default=0.003,
        help="minimum box size in cubic meters",
    )

    args = parser.parse_args()

    return parser, args
