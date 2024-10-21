# YOLO segmentation with decoding on host

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
yolo export model=yolov11{n,s}-seg.pt format=onnx imgsz=640 half=True dynamic=False simplify=True batch=1
```

#### **YOLOv9**

**1. Installation:**
```bash=
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9/
pip install -r requirements.txt
```

**2. Export:**
```bash=
python3 export.py --weights gelan-c-seg.pt --include onnx --imgsz 640 --batch-size 1 --simplify
```

### 3. Convert ONNX models to blob

The conversion from ONNX to blob is made by means of Luxonis [Blob Converter Tool](http://blobconverter.luxonis.com).


## Usage

#### Inference with YOLO8, YOLOv9 or YOLO11

```bash=
python3 main_yoloV8V9V11.py --blob <path_to_blob_file> --conf <confidence_threshold> --iou <iou_threshold>
```

#### Inference with YOLOv5

```bash=
python3 main_yoloV5.py --blob <path_to_blob_file> --conf <confidence_threshold> --iou <iou_threshold>
```
