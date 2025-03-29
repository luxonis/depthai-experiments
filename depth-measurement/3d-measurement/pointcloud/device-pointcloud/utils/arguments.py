import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "This demo uses a custom NN model approach to run custom logic - depth to pointcloud conversion - on the OAK camera itself."
    )

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
