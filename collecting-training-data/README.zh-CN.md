[英文文档](README.md)

# 训练数据收集脚本

该脚本允许使用DepthAI创建训练数据集，其中的每个条目都将包含存储的left，right，rgb和视差帧。

## 先决条件

1. 购买DepthAI模型 (请参考 [shop.luxonis.com](https://shop.luxonis.com/))
2. 安装依赖
   ```
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
参数: main.py [-h] [-p PATH] [-d] [-nd] [-m TIME] [-af {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}]

optional arguments:
  -h, --help            显示此帮助消息并退出
  -p PATH, --path PATH  存储捕获数据的路径
  -d, --dirty           允许目标路径不为空
  -nd, --no-debug       不显示调试输出
  -m TIME, --time TIME  X秒后完成执行
  -af {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}, --autofocus {AF_MODE_AUTO,AF_MODE_MACRO,AF_MODE_CONTINUOUS_VIDEO,AF_MODE_CONTINUOUS_PICTURE,AF_MODE_EDOF}
                        设置RGB相机的自动对焦模式

```

使用默认值，将数据集存储在`data`目录中

```
python3 main.py
```
