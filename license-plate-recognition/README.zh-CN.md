[英文文档](README.md)

# 车牌识别

本示例演示了如何使用Gen2 Pipeline Builder在DepthAI上运行两阶段推理。首先，在图像上检测到牌照，然后将裁剪的牌照帧发送到文本检测网络，该网络尝试对牌照文本进行解码


> :warning: **该演示进行了调整，以检测和识别中国车牌！** 它可能无法与其他车牌一起使用

## 演示

[![Gen2 License Plates recognition](https://user-images.githubusercontent.com/5244214/111202991-c62f3980-85c4-11eb-8bce-a3c517abeca1.gif)](https://www.youtube.com/watch?v=tB_-mVVNIro "License Plates recognition on DepthAI")

## 先决条件

1. 购买DepthAI模型(请参见 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
```
python -m pip install -r requirements.txt
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

### 使用设备运行程序

```
python main.py -cam
```

### 使用视频运行程序
   
```   
python main.py -vid <path>
```

按"q"退出程序。