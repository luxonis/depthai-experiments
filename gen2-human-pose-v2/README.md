# Human Pose Estimation with DepthAI

This project provides a script for human pose estimation using OpenCV, DepthAI, and a pre-trained deep learning model. The script identifies keypoints on a human body and forms pose pairs to construct the human pose.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Script Details](#script-details)

## Installation

### Requirements

- Python 3.x
- OpenCV
- DepthAI
- NumPy
- blobconverter

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/human-pose-estimation-depthai.git
   cd human-pose-estimation-depthai
   ```

2. **Install dependencies:**

   ```bash
   pip install opencv-python depthai numpy blobconverter
   ```

## Usage

1. **Run the script:**

   ```bash
   python3 main.py --model <model_name_or_path> --video <path_to_video_file>
   ```

   - `--model`: The model name or path for inference (default is `human-pose-estimation-0001`).
   - `--video`: The path to the video file (default is `keypoint_detection/pexels_walk_1.mp4`).

## Script Details

### Overview

The script captures video frames, sends them to the DepthAI device for inference, and then processes the results to display detected keypoints and pose pairs on the video frames.

### Key Components

- **Imports and Setup:**

  - Importing necessary libraries such as `cv2`, `depthai`, `numpy`, and others.
  - Defining global variables and constants like `W`, `H`, `colors`, and `POSE_PAIRS`.

- **Utility Functions:**

  - `show(frame)`: Draws keypoints and pose pairs on the frame.
  - `decode_thread(in_queue)`: Processes the neural network output to extract keypoints and pose pairs.

- **Main Execution:**

  - Parsing command-line arguments.
  - Setting up the DepthAI pipeline and device.
  - Capturing video frames and performing inference.
  - Displaying the results.

### Detailed Functions

- **show(frame):**

  Displays keypoints and pose pairs on the video frame using OpenCV drawing functions.

- **decode_thread(in_queue):**

  Processes the neural network output to decode heatmaps and part affinity fields (PAFs), and extracts keypoints and pose pairs.

### Command-line Arguments

- `-m`, `--model`: Specify the model name or path for inference.
- `-v`, `--video`: Specify the path to the video file to be processed.

---

This script is based on the `gen2-human-pose` example from the [DepthAI experiments repository](https://github.com/luxonis/depthai-experiments).
