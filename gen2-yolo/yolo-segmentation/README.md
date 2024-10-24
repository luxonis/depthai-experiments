# YOLO segmentation with decoding on host

This example shows how to perform instance segmentation on OAK devices using YOLO models ([YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLOv9](https://github.com/WongKinYiu/yolov9) and [YOLO11](https://docs.ultralytics.com/models/yolo11)). The decoding of the models' output is done on the host side. The ONNX models were exported from pretrained weights on COCO and then were converted to blob to be compatible with OAK devices.

![On-host decoding YOLO segmentation in OAK](docs/oak_segmentation_example.gif)

## Pre-requisites

### 1. Install requirements

```bash=
python3 -m pip install -r requirements.txt
```

### 2. Convert/export models to ONNX format

You can either train your custom model or try directly with a model trained on the COCO dataset. The latter is the case handled in this experiment.

#### **YOLOv5**

**1. Installation:**
```bash=
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
pip install -r requirements.txt
```

**2. Export:**
```bash=
python3 export.py --weights yolov5{n,s}-seg.pt --include onnx --imgsz 640 --opset 16 --simplify
```

#### **YOLOv8** and **YOLO11**

**1. Installation:**
```bash=
pip install ultralytics
```

**2. Export:**
```bash=
yolo export model=yolov8{n,s}-seg.pt format=onnx imgsz=640 half=True dynamic=False simplify=True batch=1
yolo export model=yolo11{n,s}-seg.pt format=onnx imgsz=640 half=True dynamic=False simplify=True batch=1
```

#### **YOLOv9**

**1. Installation:**
```bash=
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9/
pip install -r requirements.txt
```

**2. Download weights:**
```bash=
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt
```

**3. Export:**
```bash=
python3 export.py --weights gelan-c-seg.pt --include onnx --imgsz 640 --batch-size 1 --simplify
```

### 3. Convert ONNX models to blob

The conversion from ONNX to blob is made by means of Luxonis [Blob Converter Tool](http://blobconverter.luxonis.com). Note that the mean values of the ``Model optimizer parameters`` in the ``Advanced options`` must be changed from the default ``[127.5, 127.5, 127.5]`` to ``[0, 0, 0]``.


## Usage

#### Inference with YOLOv5, YOLO8, YOLOv9 or YOLO11

```bash=
python3 main.py --blob <path_to_blob_file> --conf <confidence_threshold> --iou <iou_threshold> --version <yolo_version>
```

Options:
* --blob: Path to YOLO blob file for inference. ``str``
* --conf: Set the confidence threshold. Default: 0.3. ``float``
* --iou: Set the NMS IoU threshold. Default: 0.5. ``float``
* --version: Set the YOLO version to consider. ``int``
* --input-shape: Set the input shape of YOLO model. Default: [640, 640]. ``int [int ...]``
* --fps: Set the FPS. Default: 30. ``int``
