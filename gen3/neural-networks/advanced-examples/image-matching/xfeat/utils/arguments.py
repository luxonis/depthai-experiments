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
        "-m",
        "--model",
        help="The XFeat model. Get it from the Luxonis HubAI.",
        required=False,
        default="luxonis/xfeat:mono-240x320",
        type=str,
    )

    args = parser.parse_args()

    return parser, args