## Text blurring

This example shows an implementation of [Text Detection](https://github.com/MhLiao/DB) using DepthAI SDK.
ONNX model is taken
from [PINTO's Model ZOO](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/145_text_detection_db), and reexported
with preprocessing flags.

Supported input shape is 480x640 (HxW), with expected ~3 FPS.

![Image example](imgs/example.gif)

Cat image is taken from [here](https://www.pexels.com/photo/grey-kitten-on-floor-774731/), dog image is taken
from [here](https://www.pexels.com/photo/brown-and-white-american-pit-bull-terrier-with-brown-costume-825949/).

## Pre-requisites

Install requirements:

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```