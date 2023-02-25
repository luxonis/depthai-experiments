[英文文档](README.md)

# 疲劳检测

该示例演示了Gen2 Pipeline Builder运行的 [面部检测网络](https://docs.openvinotoolkit.org/2019_R1/_face_detection_retail_0004_description_face_detection_retail_0004.html)和头部检测网络

## 演示:

![Fatigue detection](assets/fatigue.gif)

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