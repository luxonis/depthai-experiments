[英文文档](README.md)

## 主机端WLS过滤

### 背景

这给出了一个使用DepthAI Gen2 API中的`rectified_right`和`depth`流进行主机端WLS过滤的示例。 

在OAK-D上运行的示例:

![image](https://user-images.githubusercontent.com/32992551/110709334-44e93880-81b9-11eb-8901-59b7381a49c6.png)

### 怎么运行

运行

```
./install_dependencies.sh
./main.py
```

### 故障排除:

如果在运行时看到以下内容 `main.py`:
```
Traceback (most recent call last):
  File "/Users/leeroy/depthai-experiments/wls-filter/./main.py", line 52, in <module>
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(lr_check)
AttributeError: module 'cv2.cv2' has no attribute 'ximgproc'
```

这意味着 `opencv-contrib-python` 您的计算机上缺少（或需要重新安装）。请使用安装它 `python3 -m pip install opencv-contrib-python` ，然后重新运行 `./main.py`:

![image](https://user-images.githubusercontent.com/32992551/104220890-628a6380-53fd-11eb-9098-ffefc3dd3aa6.png)


