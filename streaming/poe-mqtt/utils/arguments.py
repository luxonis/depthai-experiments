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
        "--fps_limit",
        help="FPS limit.",
        required=False,
        default=None,
        type=int,
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
        "-b",
        "--broker",
        help="MQTT broker address.",
        required=False,
        default="test.mosquitto.org",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="MQTT broker port.",
        required=False,
        default=1883,
        type=int,
    )

    parser.add_argument(
        "-t",
        "--topic",
        help="MQTT topic to publish to.",
        required=False,
        default="test_topic/detections",
        type=str,
    )

    parser.add_argument(
        "-u",
        "--username",
        help="MQTT broker username.",
        required=False,
        default="",
        type=str,
    )

    parser.add_argument(
        "-pw",
        "--password",
        help="MQTT broker password.",
        required=False,
        default="",
        type=str,
    )

    args = parser.parse_args()

    return parser, args
