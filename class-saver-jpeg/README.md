[中文文档](README.zh-CN.md)

# Class Saver (JPEG)

This example demonstrates how to run MobilenetSSD and collect images of detected objects, grouped by detection label.
After running this script, DepthAI will start MobilenetSSD, and whenever it detects objects, it will add a dataset entry

Dataset is stored under `data` directory, and a main dataset file is located under `data.dataset.csv`.
For each detected object, a new dataset entry is created, with each entry having files with a precise purpose:
- `timestamp` uses `int(time.time() * 10000)` to store a timestamp of the capture. **Please note** that this value can be duplicated, if multiple objects are detected on a single image
- `label` is a human-readable label of the detected object
- `left`, `top`, `right`, `top` are object bounding box coordinates
- `raw_frame` represents a path to raw RGB frame captured on DepthAI when detection occured

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018096-47163c80-67a0-11eb-88f6-c67fb3c2f421.jpg)
  
- `overlay_frame` represents a path to RGB frame with detection overlays (bounding box and label)

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018179-63b27480-67a0-11eb-8423-4fd311a6d860.jpg)

- `cropped_frame` represents a path to cropped RGB frame containing only ROI of the detected object

   ![raw_frame example](https://user-images.githubusercontent.com/5244214/107018256-7dec5280-67a0-11eb-964e-2cc08b6b75fd.jpg)

An example entries in `dataset.csv` are shown below

```
timestamp,label,left,top,right,bottom,raw_frame,overlay_frame,cropped_frame
16125187249289,bottle,0,126,79,300,data/raw/16125187249289.jpg,data/bottle/16125187249289_overlay.jpg,data/bottle/16125187249289_cropped.jpg
16125187249289,person,71,37,300,297,data/raw/16125187249289.jpg,data/person/16125187249289_overlay.jpg,data/person/16125187249289_cropped.jpg
16125187249653,bottle,0,126,79,300,data/raw/16125187249653.jpg,data/bottle/16125187249653_overlay.jpg,data/bottle/16125187249653_cropped.jpg
16125187249653,person,71,36,300,297,data/raw/16125187249653.jpg,data/person/16125187249653_overlay.jpg,data/person/16125187249653_cropped.jpg
16125187249992,bottle,0,126,80,300,data/raw/16125187249992.jpg,data/bottle/16125187249992_overlay.jpg,data/bottle/16125187249992_cropped.jpg
16125187249992,person,71,37,300,297,data/raw/16125187249992.jpg,data/person/16125187249992_overlay.jpg,data/person/16125187249992_cropped.jpg
16125187250374,person,37,38,300,299,data/raw/16125187250374.jpg,data/person/16125187250374_overlay.jpg,data/person/16125187250374_cropped.jpg
16125187250769,bottle,0,126,79,300,data/raw/16125187250769.jpg,data/bottle/16125187250769_overlay.jpg,data/bottle/16125187250769_cropped.jpg
16125187250769,person,71,36,299,297,data/raw/16125187250769.jpg,data/person/16125187250769_overlay.jpg,data/person/16125187250769_cropped.jpg
16125187251120,bottle,0,126,80,300,data/raw/16125187251120.jpg,data/bottle/16125187251120_overlay.jpg,data/bottle/16125187251120_cropped.jpg
16125187251120,person,77,37,300,298,data/raw/16125187251120.jpg,data/person/16125187251120_overlay.jpg,data/person/16125187251120_cropped.jpg
16125187251492,bottle,0,126,79,300,data/raw/16125187251492.jpg,data/bottle/16125187251492_overlay.jpg,data/bottle/16125187251492_cropped.jpg
16125187251492,person,74,38,300,297,data/raw/16125187251492.jpg,data/person/16125187251492_overlay.jpg,data/person/16125187251492_cropped.jpg
```

## Demo

[![Gen2 Class Saver (JPEG)](https://user-images.githubusercontent.com/5244214/106964520-83b34b00-6742-11eb-8729-eff0a7584a46.gif)](https://youtu.be/gKawPaUcTi4 "Class Saver (JPEG) on DepthAI")

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```

> The dataset will be stored inside `data` directory
