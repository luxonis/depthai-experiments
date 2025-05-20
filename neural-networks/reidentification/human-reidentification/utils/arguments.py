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
        "-id",
        "--identify",
        help="Whether to run pose or face reidentification.",
        required=False,
        default="pose",
        choices=["pose", "face"],
        type=str,
    )

    parser.add_argument(
        "-cos",
        "--cos_similarity_threshold",
        help="Cosine similarity between object embeddings above which detections are considered as belonging to the same object.",
        required=False,
        default=None,
        type=float,
    )

    args = parser.parse_args()

    return parser, args
