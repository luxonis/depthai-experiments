[英文文档](README.md)

# COVID-19 口罩检测


通过本实验，您可以运行通过Google Colab教程[在此处](https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks#covid-19-maskno-mask-training-)培训的COVID-19遮罩/无遮罩对象检测器。.

您可以利用此笔记本并添加自己的训练图像以提高此遮罩检测器的质量。
我们的培训是在周末快速进行的。 

## 演示

[![COVID-19 mask-no-mask megaAI](https://user-images.githubusercontent.com/5244214/112673778-6a3a9f80-8e65-11eb-9b7b-e352beffe67a.gif)](https://youtu.be/c4KEFG2eR3M "COVID-19 mask detection")

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
