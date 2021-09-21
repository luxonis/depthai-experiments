## Installation

```
python3 -mpip install -r requirements.txt
```

## Creating blobs

When you run any of the scripts in this folder, it will automatically export PyTorch model into `.onnx` format, simplify it (if needed), and convert it to `.blob` with `blobconverter` package.
