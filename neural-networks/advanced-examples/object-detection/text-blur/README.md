## [Gen2] Text blurring on DepthAI

This example shows an implementation of [Text Detection](https://hub.luxonis.com/ai/models/131d855c-60b1-4634-a14d-1269bb35dcd2?view=page) on DepthAI in the Gen3 API system with additional text blurring.

![Image example](imgs/example.gif)

Cat image is taken from [here](https://www.pexels.com/photo/grey-kitten-on-floor-774731/), dog image is taken from [here](https://www.pexels.com/photo/brown-and-white-american-pit-bull-terrier-with-brown-costume-825949/).

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py [options]
```

Options:

* `-bt, --box_thresh`: Set the box confidence threshold. Default: *0.2*.
* `-t, --thresh`: Set the bitmap threshold. Default: *0.1*.
* `-ms, --min_size`: Set the minimum size of box (area). Default: *1*.
* `-mc, --max_candidates`: Maximum number of returned box candidates. Default: *75*.
