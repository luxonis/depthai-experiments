## LaneNet on DepthAI

This example shows an implementation of [LaneNet](https://arxiv.org/pdf/1802.05591.pdf) on DepthAI.  Model is pretrained on TuSimple dataset. It is taken from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/141_lanenet-lane-detection) and converted with model optimizer so that no additional preprocessing is needed.

Post-processing is not the same as in official paper!
Please refer to https://arxiv.org/abs/1802.05591 for how postprocessing should be done!

The inference speed is around 2-4 FPS

![Image example](imgs/example.gif)

Example video is taken from [here](https://github.com/udacity/CarND-LaneLines-P1).

### Output and postprocessing

The model produces two outputs - *(1)* binary segmentation mask where 1 represents line and *(2)* embeddings of dimension 4 for each pixel. Note that our post-processing differs from the original post-processing in [LaneNet](https://arxiv.org/pdf/1802.05591.pdf), and was made just to showcase the model on the DepthAI. 

We currently perform DBSCAN clustering of embeddings, which were sorted using Numpy's lexsort. This works OK and lanes are usually allocated to the same cluster during different frames, as long as no new line is detected on the left side of the leftmost line. For proper post-processing please refer to [LaneNet](https://arxiv.org/pdf/1802.05591.pdf) paper.

## Installation

```
python3 -m pip install -r requirements.txt
python3 download.py
```

## Usage

Run the application

```
python3 main.py

arguments:
  -h, --help            show this help message and exit
  -v VIDEO_PATH, --video VIDEO_PATH
                        Path to video to use for inference. Default: vids/vid3.mp4
```
