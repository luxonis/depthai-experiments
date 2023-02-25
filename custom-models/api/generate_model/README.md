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

#### Cosinus distance

[Cosinus distance/similarity](https://en.wikipedia.org/wiki/Cosine_similarity) model that allows you to compare vectors on the device itself. Currently it's not being used anywhere (closed [PR here](https://github.com/luxonis/depthai-experiments/pull/259)), but it could be used in [Face recognition](https://github.com/luxonis/depthai-experiments/tree/master/face-recognition) / [Person reidentification](https://github.com/luxonis/depthai-experiments/tree/master/pedestrian-reidentification) demos.

Creating this model was a bit tricky, as some values in between calculations exceeded [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples) upper limit of 65504 (2^16). That means it calculated `inf` when some multiplication values were too large, which lead to `0` or `inf` end result.
**Solution:** we divide values in between the calculation by 1000, so no value exceeds the FP16 upper value limit.

It would also be possible to have many inputs (for eg. multiple different people vectors) and then compare the new vector with all of them at once to decrease computation time. [Multi-Input Frame Concationation](https://docs.luxonis.com/projects/api/en/latest/samples/NeuralNetwork/concat_multi_input) example would be a useful reference.