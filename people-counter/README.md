# People counting

One of the basic examples of DepthAI usage - counting people in front of the camera and save the
results to JSON file (for further processing)

App can be useful as a starting point to other applications or to monitor e.x. the conference
rooms usage

## Pre-requisites

1. Purchase a DepthAI model (see https://shop.luxonis.com/)
2. Install DepthAI API (see [here](https://docs.luxonis.com/api/) for your operating system/platform)

## Install project requirements

```
python3 -m pip install -r requirements.txt
```

## Run this example

```
python3 main.py
```

By default, you should see a debug window and console output that shows how many people were
detected. Also, you should see `results.json` file with timestamped results.

If you want to run it without preview, just to collect the data, you can modify `main.py` and set

```diff
- debug = True
+ debug = False
```

Then the app will run without preview window nor debug messages, will just save the results to `results.json`