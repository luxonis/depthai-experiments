## [Gen2] Deeplabv3 on DepthAI - depth cropping

This example shows how to run Deeplabv3+ on DepthAI in the Gen2 API and crop the depth image based on the models output.

[![Semantic Segmentation on DepthAI](https://user-images.githubusercontent.com/18037362/120105024-29a21f80-c14f-11eb-97d7-5d59e144b19d.gif)](https://www.youtube.com/watch?v=M1LTqGy-De4 "Deeplabv3")

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