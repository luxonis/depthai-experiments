## Blur Faces in Real Time

This example shows how to blur detected faces in real-time. Face blurring happens on the host, with the help of OpenCV. You could also create
a custom CV model ([tutorial](https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/)) that would blur the frame based on the bounding boxes.

## Demo

![Blur Face](https://user-images.githubusercontent.com/18037362/139135932-b907f037-9336-4c42-a479-5715d9693c9c.gif)

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
python3 main.py
```

