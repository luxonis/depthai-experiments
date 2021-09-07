# Gen2 Play Encoded Stream

This application plays the h265 encoded stream on the host machine.

It does so by piping the output stream of the device to the input of the ffplay process.

## Demo

![Simple demo](https://user-images.githubusercontent.com/59799831/132342531-f9a4fc02-e8a8-4aca-85c2-0fd610aa7249.gif)

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
