# Cumulative Object Counting

![cumulative object counting](https://raw.githubusercontent.com/TannerGilbert/Tensorflow-2-Object-Counting/master/doc/cumulative_object_counting.PNG)

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
# DepthAI SDK
python3 main.py
```

```bash
# DepthAI API
## Camera example
python main.py -m models/mobilenet-ssd.blob
## Video example
python main.py -m models/mobilenet-ssd.blob -v demo/example_01.mp4 -a  
```

Arguments:
```
usage: main.py [-h] [-m MODEL] [-v VIDEO_PATH] [-roi ROI_POSITION] [-a] [-sh] [-sp SAVE_PATH] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        File path of .blob file. (default: models/mobilenet-ssd.blob)
  -v VIDEO_PATH, --video_path VIDEO_PATH
                        Path to video. If empty OAK-RGB camera is used. (default='') (default: )
  -roi ROI_POSITION, --roi_position ROI_POSITION
                        ROI Position (0-1) (default: 0.5)
  -a, --axis            Axis for cumulative counting (default=x axis) (default: True)
  -sh, --show           Show output (default: True)
  -sp SAVE_PATH, --save_path SAVE_PATH
                        Path to save the output. If None output won't be saved (default: )
  -s, --sync            Sync RGB output with NN output (default: False)
```

## Inspired by / Based on
* [Tensorflow 2 Object Counting](https://github.com/TannerGilbert/Tensorflow-2-Object-Counting)
* [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) 
* [tensorflow_object_counting_api](https://github.com/ahmetozlu/tensorflow_object_counting_api)