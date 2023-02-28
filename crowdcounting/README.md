##  Crowd Counting with density maps on DepthAI

This example shows an implementation of Crowd Counting with density maps on DepthAI in the Gen2 API system.  We use [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) ([LICENSE](https://github.com/cvlab-stonybrook/DM-Count/blob/master/LICENSE)) model, which has a VGG-19 backbone and is trained on Shanghai B data set.

The model produces density map from which predicted count can be computed.

Input video is resized to 426 x 240 (W x H). Due to a relatively heavy model, the inference speed is around 1 FPS.

![Image example](assets/example.gif)

![image](https://user-images.githubusercontent.com/32992551/171780142-5cd4f2a4-6c51-4dbc-9e3e-17062a9c6c6c.png)


Example shows input video with overlay density map input. Example video taken from [VIRAT](https://viratdata.org/) dataset.

## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

1. Download sample videos
   ```
   python3 download.py
   ```
2. Install requirements
   ```
   python3 -m pip install -r requirements.txt
   ```

### Launch the script

```
# DepthAI API
python3 main.py [options]

# DepthAI SDK
python3 main.py
```

Options:

* -v, --video_path: Path to the video input for inference. Default: *vids/virat.mp4*.
* -nn, --nn_model: Select model path for inference. Default: *models/vgg_openvino_2021.4_6shave.blob*
