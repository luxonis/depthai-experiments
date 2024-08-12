# DeeplabV3 on DepthAI

This example shows how to run DeeplabV3+ on DepthAI.

There are three available blobs for different modelsand input sizes:

* `deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob`, for 256x256 input multiclass segmentation.
* `deeplab_v3_mnv2_256x256`, for 256x256 input person segmentation.
* `deeplab_v3_mnv2_513x513`, for 513x513 input person segmentation.

See the example in action detecting monitor and a person:
![Semantic multiclass segmentation on DepthAI](imgs/example.gif)
![Semantic person segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/109359126-25a9ed00-7842-11eb-9071-cddc7439e3ca.png)(https://www.youtube.com/watch?v=zjcUChyyNgI "Deeplabv3+ Custom Training for DepthAI")

The multiclass model has a MobilenetV2 backbone and is trained on Pascal VOC 2012 dataset. It contains the following classes:

- *Person:* person
- *Animal:* bird, cat, cow, dog, horse, sheep
- *Vehicle:* aeroplane, bicycle, boat, bus, car, motorbike, train
- *Indoor:* bottle, chair, dining table, potted plant, sofa, tv/monitor

You can find the tutorial for training the model and generation of *.blob* file [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks) - **DeepLabV3plus_MNV2.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/DeepLabV3plus_MNV2.ipynb). You can create a copy of the Colab Notebook and try training the model on your own!

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application.

```
python3 main.py

optional arguments:
  -h, --help            show this help message and exit
  -nn {multiclass, person_small, person_large}, --nn-choice {multiclass, person_small, person_large}
                        Choose the neural network model used for segmentation (multiclass is default)
  -cam {left, rgb, right}, --cam-input {left, rgb, right}
                        Choose camera for inference source (rgb is default)
```
