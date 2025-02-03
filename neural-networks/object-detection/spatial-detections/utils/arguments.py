import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser()

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
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=30.0,
        type=float,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model reference to use for object detection.",
        required=False,
        default="luxonis/yolov6-nano:r2-coco-512x288",
        type=str,
    )

    args = parser.parse_args()

    return parser, args
