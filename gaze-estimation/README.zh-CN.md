[英文文档](README.md)

#  凝视估计

本示例演示了如何使用Gen2 Pipeline Builder在DepthAI上运行3阶段推理(3序列，2并行)。

[![Gaze Example Demo](https://user-images.githubusercontent.com/5244214/106155937-4fa7bb00-6181-11eb-8c23-21abe12f7fe4.gif)](https://user-images.githubusercontent.com/5244214/106155520-0f483d00-6181-11eb-8b95-a2cb73cc4bac.mp4)


原OpenVINO演示，在其上做这个例子，是官方[在这里](https://docs.openvinotoolkit.org/2021.1/omz_demos_gaze_estimation_demo_README.html), 从英特尔，并与NCS2实现漂亮的图表和说明， [在这里](https://github.com/LCTyrell/Gaze_pointer_controller) @LCTyrell。

![graph](https://user-images.githubusercontent.com/32992551/103378235-de4fec00-4a9e-11eb-88b2-621180f7edef.jpeg)

图: @LCTyrell

## 先决条件

1. 购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装条件
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
用法: main.py [-h] [-nd] [-cam] [-vid VIDEO]

可选参数:
   -h, --help            显示此帮助消息并退出
   -nd, --no-debug       禁止调试输出
   -cam, --camera        使用DepthAI 4K RGB相机进行推理(与-vid冲突)
   -vid VIDEO, --video VIDEO
                         用于推理的视频文件的路径(与-cam冲突)
```

要与视频文件一起使用，请使用以下参数运行脚本

```
python3 main.py -vid ./demo.mp4
```

要与DepthAI 4K RGB相机一起使用，请改用

```
python3 main.py -cam
``` 
