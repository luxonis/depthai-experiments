import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-det",
        "--det_model",
        help="Detection model HubAI reference.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-rec",
        "--rec_model",
        help="Recognition model HubAI reference.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-cos",
        "--cos_similarity_threshold",
        help="Cosine similarity between object embeddings above which detections are considered as belonging to the same object.",
        required=False,
        default=0.5,
        type=float,
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
        help="FPS limit for the model runtime.",
        required=False,
        default=30.0,
        type=float,
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
