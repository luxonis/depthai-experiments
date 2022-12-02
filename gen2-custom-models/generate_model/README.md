## Installation

```
python3 -mpip install -r requirements.txt
```

## Creating blobs

When you run any of the scripts in this folder, it will automatically export PyTorch model into `.onnx` format, simplify it (if needed), and convert it to `.blob` with `blobconverter` package.

## Models

#### Concatenate frames

![Concat frames](https://user-images.githubusercontent.com/18037362/134209980-09c6e2f9-8a26-45d5-a6ad-c31d9e2816e1.png)

Uses [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html) operation to concatenate multiple frames.

#### Blur frames

![Blur frames](https://docs.luxonis.com/en/latest/_images/blur.jpeg)

Uses [kornia.filters.GaussianBlur2d](https://kornia.readthedocs.io/en/latest/filters.html?highlight=GaussianBlur2d#kornia.filters.GaussianBlur2d) filter to create blurring model.

#### Corner detection

![Laplacian corner detection](https://user-images.githubusercontent.com/18037362/134209951-4e1c7343-a333-4fb6-bdc9-bc86f6dc36b2.jpeg)

Uses [kornia.filters.Laplacian](https://kornia.readthedocs.io/en/latest/filters.html?highlight=laplacian#kornia.filters.Laplacian) filter to create corner detection model.