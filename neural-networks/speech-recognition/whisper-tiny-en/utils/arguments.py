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
        "-a",
        "--audio_file",
        type=str,
        required=False,
        default="assets/audio_files/command_LED_yellow.mp3",
        help="The path to the audio file to process",
    )

    args = parser.parse_args()

    return parser, args
