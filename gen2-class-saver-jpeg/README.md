# Gen2 Class Saver (JPEG)

This example demonstrates how to run MobilenetSSD and collect images of detected objects, grouped by detection label.
After running this script, DepthAI will start MobilenetSSD, and whenever it detects an object, it will crop it from 
frame and save inside `data/<label>` directory, with `<timestamp>.jpeg` filename


## Demo

[![Gen2 Class Saver (JPEG)](https://user-images.githubusercontent.com/5244214/106964520-83b34b00-6742-11eb-8729-eff0a7584a46.gif)](https://youtu.be/gKawPaUcTi4 "Class Saver (JPEG) on DepthAI")

## Pre-requisites

1. Purchase a DepthAI model (see [shop.luxonis.com](https://shop.luxonis.com/))
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

## Usage

```
python3 main.py
```

The dataset will be stored inside `data` directory
