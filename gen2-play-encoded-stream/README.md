# Gen2 Play Encoded Stream

This application plays the h265 encoded stream on the host machine.

It does so by piping the output stream of the device to the input of the ffplay process.

## Demo

![Encoding demo](https://user-images.githubusercontent.com/59799831/132475640-6e9f8b7f-52f4-4f75-af81-86c7f6e45b94.gif)

## Pre-requisites

Purchase a DepthAI model (see https://shop.luxonis.com/)

## Install project requirements

```
sudo apt install ffmpeg
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```
