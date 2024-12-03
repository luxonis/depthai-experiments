import argparse
import cv2
import numpy as np

import depthai as dai
from depthai_nodes.parsing_neural_network import ParsingNeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nn",
    "--neural-network",
    type=str,
    choices=[
        "midas_small",
        "midas_medium",
        "midas_large",
        "midas_xlarge",
        "midas_xxlarge",
    ],
    default="midas_small",
    help="Choose the neural network model used for depth estimation (midas_small is default)",
)
parser.add_argument(
    "-fps",
    "--frames-per-second",
    type=int,
    default=10,
    help="Choose the number of frames per second (10 is default)",
)
args = parser.parse_args()

nn_configs = {
    "midas_small": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-192x256",
        "size": (256, 192),
    },
    "midas_medium": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-256x384",
        "size": (384, 256),
    },
    "midas_large": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-288x512",
        "size": (512, 288),
    },
    "midas_xlarge": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-384x512",
        "size": (512, 384),
    },
    "midas_xxlarge": {
        "model_slug": "midas-v2-1",
        "model_version_slug": "small-768x1024",
        "size": (1024, 768),
    },
}

model_config = nn_configs[args.neural_network]
model_slug = model_config["model_slug"]
model_version_slug = model_config["model_version_slug"]

FPS = args.frames_per_second

model_description = dai.NNModelDescription(
    modelSlug=model_slug,
    modelVersionSlug=model_version_slug,
)

#archive_path = dai.getModelFromZoo(model_description)
#nn_archive = dai.NNArchive(archive_path)

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    # Set up camera
    camera_node = pipeline.create(dai.node.Camera).build()

    # Set up neural network with parser and link it to camera output
    nn_with_parser_node = pipeline.create(ParsingNeuralNetwork).build(
        camera_node, 
        model_description,  
        fps=FPS
    )

    parser_output_queue = nn_with_parser_node.out.createOutputQueue()
    frameQueue = nn_with_parser_node.passthrough.createOutputQueue()

    # Start pipeline
    pipeline.start()

    while pipeline.isRunning():

        inImage: dai.ImgFrame = frameQueue.get()
        frame = inImage.getCvFrame()

        parsed_output = parser_output_queue.get()

        depth_map = parsed_output.map
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        depth_map_colored = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_JET)

        visualization = np.concatenate((frame, depth_map_colored), axis=1)
       
        cv2.imshow("img", visualization)
        if cv2.waitKey(1) == ord("q"):
            break
