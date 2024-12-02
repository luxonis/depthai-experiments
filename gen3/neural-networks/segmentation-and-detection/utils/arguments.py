import argparse
from typing import Tuple
import os

def initialize_argparser():
    
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser()
    parser.description = "Example script to run any detection + segmentation model available in HubAI on OAK devices. \
        All you need is a model slug of the model and the script will download the model from HubAI and create \
        the whole pipeline with visualizations. You also need an OAK device connected to your computer. \
        If using OAK-D Lite, please set the FPS limit to 28."

    parser.add_argument(
        "-m",
        "--model_slug",
        help="Slug of the model copied from HubAI.",
        required=True,
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
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default="",
        type=str,
    )
    args = parser.parse_args()

    return parser, args