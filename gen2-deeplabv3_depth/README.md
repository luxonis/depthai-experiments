## [Gen2] Deeplabv3 on DepthAI - depth cropping

This example shows how to run Deeplabv3+ on DepthAI in the Gen2 API and crop the depth image based on the models output.

![Deeplabv3 Depth GIF](https://user-images.githubusercontent.com/59799831/132396685-c494f21b-8101-4be4-a787-dd382ae6b470.gif)

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