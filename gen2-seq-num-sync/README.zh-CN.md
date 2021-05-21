[英文文档](README.md)

# Gen2 帧同步

本示例演示如何使用序列号同步传入的帧。这样可以显示在同一时刻拍摄的帧

## 演示

![demo](./demo.png)

## 先决条件

1. 购买DepthAI(请参见 [淘宝](https://item.taobao.com/item.htm?id=626257175462))
2. 安装依赖
   ```
   python3 -m pip install -r requirements.txt
   ```

## 用法

```
python3 main.py
```

## TODO

- 添加彩色摄像机流（等待序列号在RGB和Mono摄像机之间同步）