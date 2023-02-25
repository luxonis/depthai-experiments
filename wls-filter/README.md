[中文文档](README.zh-CN.md)

# Host-Side WLS Filtering

## Background

This gives an example of doing host-side WLS filtering using the `rectified_right` and `depth` stream from DepthAI Gen2 API.  

Example of running on OAK-D:
![image](https://user-images.githubusercontent.com/32992551/110709334-44e93880-81b9-11eb-8901-59b7381a49c6.png)

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

## Troubleshooting

If you see the following when running `main.py`:
```
Traceback (most recent call last):
  File "/Users/leeroy/depthai-experiments/wls-filter/./main.py", line 52, in <module>
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(lr_check)
AttributeError: module 'cv2.cv2' has no attribute 'ximgproc'
```

This means that `opencv-contrib-python` is missing (or needs to be reinstalled) on your machine.  Please install it using `python3 -m pip install opencv-contrib-python` and then re-run `./main.py`:

![image](https://user-images.githubusercontent.com/32992551/104220890-628a6380-53fd-11eb-9098-ffefc3dd3aa6.png)


