
## [Gen2] Creating custom models

This experiment can be used as a tutorial on how users can create custom models with
PyTorch/Kornia, convert them and run them on the DepthAI.

[Tutorial documentation here](https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/).

`blur.py`, `concat.py` and `edge.py` are scripts that run created custom models. `generate_model/` folder contains scripts that create these custom models (frame blurring, frame concatenation and edge detection).

## Demos

**Concatenate frames**
![Concat frames](https://docs.luxonis.com/en/latest/_images/concat_model.png)

**Blur frames**
![Blur frames](https://docs.luxonis.com/en/latest/_images/blur.jpeg)

**Corner detection**
![Laplacian corner detection](https://docs.luxonis.com/en/latest/_images/laplacian.jpeg)

## Pre-requisites

Install requirements
```
python3 -m pip install -r requirements.txt
```
