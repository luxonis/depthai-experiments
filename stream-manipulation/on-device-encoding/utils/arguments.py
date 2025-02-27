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
        default=30,
        type=int,
    )

    parser.add_argument(
        "-c",
        "--codec",
        choices=["h264", "h265", "mjpeg"],
        default="h264",
        type=str,
        help="Video encoding (h264 is default)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output file.",
        required=False,
        default="video.mp4",
        type=str,
    )

    args = parser.parse_args()

    return parser, args
