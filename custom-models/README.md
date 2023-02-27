##  Creating custom models

This experiment can be used as a tutorial on how users can create custom models with
PyTorch/Kornia, convert them and run them on the DepthAI.

[Tutorial documentation here](https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/).

`blur.py`, `concat.py` and `edge.py` are scripts that run created custom models. `generate_model/` folder contains scripts that create these custom models (frame blurring, frame concatenation and edge detection).

## Demos

### Concatenate frames

![Concat frames](https://user-images.githubusercontent.com/18037362/134209980-09c6e2f9-8a26-45d5-a6ad-c31d9e2816e1.png)

### Blur frames

![Blur frames](https://docs.luxonis.com/en/latest/_images/blur.jpeg)

### Corner detection

![Laplacian corner detection](https://user-images.githubusercontent.com/18037362/134209951-4e1c7343-a333-4fb6-bdc9-bc86f6dc36b2.jpeg)

### Difference between 2 frames

`diff.py` computes the difference between two consecutive color frames and visualizes that along with the color frame.

![diff-demo](https://user-images.githubusercontent.com/18037362/185765421-3f391d9c-341a-4e5c-9f5d-3ce322347af1.gif)


## Usage

Choose one of the following options:
```bash
# For DepthAI API
cd ./api

# For DepthAI SDK
cd ./sdk
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 $SCRIPT_NAME.py
``` 

