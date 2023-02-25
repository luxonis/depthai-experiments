# Mask R-CNN on DepthAI

This example shows how to run a Mask R-CNN model on DepthAI. Please note that since it is a heavier model so the expected speed is ~ 2-4 FPS

![Example](https://user-images.githubusercontent.com/56075061/145182204-af540962-f233-480c-82a0-56b2587e5072.gif)

![Img](https://user-images.githubusercontent.com/18037362/162579543-7ebfe41c-d6a9-45e4-aa41-36969cc21894.png)

## Export
You can export your Mask R-CNN directly to OpenVINO IR and then to blob. For the first conversion you can use [this repository](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/mo_pytorch), where you would select `torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn` model and load your weights. More instruction can be found in the repository. After obtaining XML and BIN you can use [blobconverter](https://blobconverter.luxonis.com/) to convert them to blob and use them on OAK.

## Usage

### Navigate to directory

```bash
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
