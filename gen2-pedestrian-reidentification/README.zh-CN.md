[英文文档](README.md)

# 行人识别

本示例演示了如何使用Gen2 Pipeline Builder在DepthAI上运行两阶段推理。

原OpenVINO演示，在其上实现这个例子，[在这里](https://docs.openvinotoolkit.org/2020.1/_demos_pedestrian_tracker_demo_README.html).

## 演示

[![Pedestrian Re-Identification](https://user-images.githubusercontent.com/32992551/108567421-71e6b180-72c5-11eb-8af0-c6e5c3382874.png)](https://www.youtube.com/watch?v=QlXGtMWVV18 "Person Re-ID on DepthAI")

## 先决条件

1. 购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
main.py [-h] [-nd] [-cam] [-vid VIDEO] [-w WIDTH] [-lq]
```

可选参数:
 - `-h, --help`      显示此帮助消息并退出
 - `-nd, --no-debug` 禁止调试输出
 - `-cam, --camera`  使用DepthAI RGB相机进行推理（与-vid冲突）
 - `-vid VIDEO, --video VIDEO` 用于推理的视频文件的路径（与-cam冲突）
 - `-w WIDTH, --width WIDTH` 可视宽度
 - `-lq, --lowquality` 使用调整大小的帧而不是源


要与视频文件一起使用，请使用以下参数运行脚本

```
python3 main.py -vid input.mp4
```

要与DepthAI 4K RGB相机一起使用，请改用

```
python3 main.py -cam
``` 
