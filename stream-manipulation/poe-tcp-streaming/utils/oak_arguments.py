import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(
        help="Mode of the script.", dest="mode", required=True
    )
    subparsers.add_parser("server", help="Run in server mode.")
    parser_client = subparsers.add_parser("client", help="Run in client mode.")
    parser_client.add_argument(
        "address", help="IP address of the host device.", type=str
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
        "-media",
        "--media_path",
        help="Path to the media file you aim to run the model on. If not set, the model will run on the camera input.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit.",
        required=False,
        default=30,
        type=int,
    )

    args = parser.parse_args()

    return parser, args
