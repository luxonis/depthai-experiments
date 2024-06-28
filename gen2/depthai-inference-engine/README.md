# DepthAI with Inference Engine

This example shows how you can run inference on your DepthAI device using OpenVINOs [Inference Engine](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html). This example does not use the DepthAI library (so there are no pipelines) as it runs the DepthAI device in [NCS (Neural Compute Stick) mode](https://docs.luxonis.com/en/latest/pages/faq/#what-is-ncs2-mode).

A common use case to run your model with IE (Inference Engine) first is to check if your model conversion to OpenVINOs IR format (eg. from TF/ONNX) was successful. After you run it successfully with the IE you can then proceed with [compiling the IR model](https://docs.luxonis.com/en/latest/pages/model_conversion/) into the **.blob**, which is required by the DepthAI library.

The NN model (facial cartoonization) was taken from PINTOs [model-zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/062_facial_cartoonization).

## Demo

![demo](https://user-images.githubusercontent.com/18037362/123509323-15c9da80-d675-11eb-98c9-bdee79e10664.png)

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

```
python3 main.py
```

