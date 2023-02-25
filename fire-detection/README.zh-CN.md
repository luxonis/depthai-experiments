[英文文档](README.md)

火焰检测
================

该示例演示了运行[fire detection network](https://github.com/StephanXu/FireDetector/tree/python)的Gen2 Pipeline Builder

## 演示:

![demo](api/images/fire_demo.gif)
--------------------

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

```bash
python main.py -vid <path>
```

要与DepthAI 4K RGB相机一起使用，请改用
```bash
python main.py -cam
```

> 按"q"退出程序。
