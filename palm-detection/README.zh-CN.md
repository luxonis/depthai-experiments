[英文文档](README.md)

手掌检测
================

该示例演示了Gen2 Pipeline Builder运行的[手掌检测网络](https://google.github.io/mediapipe/solutions/hands#palm-detection-model)  

## 演示:

![demo](assets/palm_detection.gif)
--------------------

## 先决条件

1. 购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```bash
   python3 -m pip install -r requirements.txt
   ```


## 用法

```bash
用法: main.py [-h] [-nd] [-cam] [-vid VIDEO]

可选参数:
  -h, --help            显示此帮助消息并退出
  -nd, --no-debug       阻止调试输出
  -cam, --camera        使用DepthAI 4K RGB相机进行推理（与-vid冲突）
  -vid VIDEO, --video VIDEO
                        用于推理的视频文件的路径（与-cam冲突）

```

要与视频文件一起使用，请使用以下参数运行脚本

```bash
python main.py -vid <path>
```

要与DepthAI 4K RGB相机一起使用，请改用
```bash
python main.py -cam
```

> 按"q"退出程序。
