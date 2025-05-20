import argparse

def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "Example script on how to run Foundation Stereo with DepthAI device. \
        You also need a DepthAI device connected to your computer. \
        Please make sure your resolution matches the model resolution."
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Path to ONNX model file.",
        default="models/foundation_stereo_640x416_32.onnx",
        type=str,
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
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=15,
        type=int,
    )

    parser.add_argument(
        "-r",
        "--resolution",
        help="Resolution of the streams, select 400 (for 640x400) or 800 (for 1280x800).",
        required=False,
        default=400,
        type=int,
    )

    args = parser.parse_args()

    return parser, args
