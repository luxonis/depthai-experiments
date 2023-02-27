[中文文档](README.zh-CN.md)

# COVID-19 mask detection


This experiment allows you to run the COVID-19 mask/no-mask object detector which was trained via the Google Colab tutorial [here](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-).

You can leverage this notebook and add your own training images to improve the quality of this mask detector.  
Our training was a quick weekend effort at it.  

## Demo

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/112673778-6a3a9f80-8e65-11eb-9b7b-e352beffe67a.gif)](https://youtu.be/c4KEFG2eR3M "COVID-19 mask detection")

## Install requirements

```
python3 -m pip install -r requirements.txt
```

## Usage


### Navigate to directory

```bash
cd ./api
```

### Launch the script

To use with a video file, run the script with the following arguments

```
python3 main.py -vid ./input.mp4
```

To use with DepthAI 4K RGB camera, use instead

```
python3 main.py -cam
``` 

Arguments:
```
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to video file to be used for inference (conflicts with -cam)
```
