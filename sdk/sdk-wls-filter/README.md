[中文文档](README.zh-CN.md)

## Host-Side WLS Filtering

### Demo

This gives an example of doing host-side WLS filtering using DepthAI SDK.

Example of running on OAK-D:
![image](https://user-images.githubusercontent.com/32992551/110709334-44e93880-81b9-11eb-8901-59b7381a49c6.png)

### Usage

Run

```
python -m pip install -r requirements.txt
python main.py
```

### Troubleshooting

If you see the following when running `main.py`:

```
Traceback (most recent call last):
  File "/Users/leeroy/depthai-experiments/wls-filter/./main.py", line 52, in <module>
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(lr_check)
AttributeError: module 'cv2.cv2' has no attribute 'ximgproc'
```

This means that `opencv-contrib-python` is missing (or needs to be reinstalled) on your machine. Please install it
using `python3 -m pip install opencv-contrib-python` and then re-run `python main.py`

![image](https://user-images.githubusercontent.com/32992551/104220890-628a6380-53fd-11eb-9098-ffefc3dd3aa6.png)


