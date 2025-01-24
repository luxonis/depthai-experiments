import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser()
    parser.description = (
        "General example script to run any model available in HubAI on DepthAI device. \
        All you need is a model slug of the model and the script will download the model from HubAI and create \
        the whole pipeline with visualizations. You also need a DepthAI device connected to your computer. \
        If using OAK-D Lite, please set the FPS limit to 28."
    )

    parser.add_argument(
        "-m",
        "--model",
        help="HubAI model reference.",
        required=True,
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
        "-ann",
        "--annotation_mode",
        help="Annotation mode. Can be either 'segmentation', 'segmentation_with_annotation', or None (default).",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
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
        "-api",
        "--api_key",
        help="HubAI API key to access private model.",
        required=False,
        default="",
        type=str,
    )

    args = parser.parse_args()

    return parser, args
