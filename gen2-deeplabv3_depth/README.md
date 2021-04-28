## [Gen2] Deeplabv3 on DepthAI - depth cropping

This example shows how to run Deeplabv3+ on DepthAI in the Gen2 API and crop the depth image based on the models output.

[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/32992551/109359126-25a9ed00-7842-11eb-9071-cddc7439e3ca.png)](https://www.youtube.com/watch?v=zjcUChyyNgI "Deeplabv3+ Custom Training for DepthAI")

## Pre-requisites

Install requirements
```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py [-nn {path}]
```

You can use a different model from the `gen2-deeplabv3_person` folder (`mvn3` or `513x513` input)