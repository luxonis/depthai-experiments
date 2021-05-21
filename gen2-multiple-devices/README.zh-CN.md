[英文文档](README.md)

# 每个主机的Gen2多台设备

本示例说明如何在单个主机上使用多个DepthAI。 该演示将找到连接到主机的所有设备，并显示每个设备的RGB预览。 在此演示中，所有设备都使用相同的管道（仅显示彩色摄像机预览）。 并非必须如此，每个设备都可以运行单独的管道。

## 演示

[![Multiple devices per host](https://user-images.githubusercontent.com/18037362/113307040-01d83c00-9305-11eb-9a42-c69c72a5dba5.gif)](https://www.youtube.com/watch?v=N1IY2CfhmEc "Multiple devices per host")

两个DepthAI互相看着对方。

## DepthAI购买渠道

购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))

## 设置

```
python3 -m pip -U pip
python3 -m pip install -r requirements.txt
```

## 运行

```
python3 main.py
```
