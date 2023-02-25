# Kalman filter
This example performs filtering of 2D bounding boxes and spatial coordinates of tracked objects using the Kalman filter.

The Kalman filter is used to obtain better estimates of objects' locations by combining measurements and past estimations. Even when measurements are noisy this filter performs quite well.

Here is a short explanation of the Kalman filter: https://www.youtube.com/watch?v=s_9InuQAx-g.

# Demo

![video](https://user-images.githubusercontent.com/69462196/197813200-236e950e-3dda-403f-b5cd-8d11f0e86124.gif)

In the demo red represents filtered data and blue unfiltered.

## Usage

### Navigate to directory

```
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py
```