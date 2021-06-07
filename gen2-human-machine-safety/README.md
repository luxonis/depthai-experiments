# Gen2 Human-Machine safety

This example demonstrates how to detect dangerous objects and calculates distance to a human hand (palm).
It uses mobilenet spatial detection netork node to get spatial coordinates of dangerous objects and palm
detection network to detect human palm. Script `palm_detection.py` handles decoding of the [palm detection network](https://google.github.io/mediapipe/solutions/hands#palm-detection-model) and returns the bounding box of a detected palm.
Instead of sending the bounding box of the detected palm back to device to the`SpatialLocationCalculator`, this example
uses function `def calc_spatials(self, bbox, depth):` to calculate spatial coordinates on the host (using bbox and depth map). After we have spatial coordiantes of both the dangerous object and the palm, we calculate the spatial distance of the two and if it's blow the threshold `WARNING_DIST`, it will output a warning.

## Demo:


## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

```bash
   python3 main.py
```

> Press 'q' to exit the program.
