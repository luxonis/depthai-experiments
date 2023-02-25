# OAK + Roboflow Demo

An app creating [Roboflow](https://roboflow.com) dataset using detections from
an [OAK-1](https://store.opencv.ai/products/oak-1) camera.

## Demo

Live preview shows MobileNet SSD detections. After pressing `enter` the app grabs frames and uploads them to Roboflow
dataset along with annotations.

https://user-images.githubusercontent.com/26127866/147658296-23be4621-d37a-4fd6-a169-3ea414ffa636.mp4

## Getting Started

1. If it's your first project with OAK, follow
   this [first steps guide](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#first-steps-with-depthai).

2. Clone repository. Install requirements. Download model.

```bash
git clone https://github.com/tomek-l/oak-roboflow-integration.git
cd oak-roboflow-integration
pip3 install -r requirements.txt --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/
cd models/
bash download.sh
cd ..
```

3. Setup Roboflow account

- Get API key ([app.roboflow.com](https://app.roboflow.com/) -> `settings` -> `workspaces` -> `Roboflow API` -> Copy
  private API key)
- Create new (empty) project at [app.roboflow.com](https://app.roboflow.com/). Then copy the project's (a.k.a.
  dataset's) name.

4. Run the code with your `API key` and `dataset name`:

```shell
cd ./api
python3 main.py --dataset oak-dataset --api_key vkIkZac3CXvp0RZ31B3f
```

- Press `enter` to capture and upload frames with annotations.
- Press `q` to exit

5. Experiment with optional cmd arguments for auto-uploading `--autoupload_threshold` and `--autoupload_interval`. For
   example, to automatically upload annotations with confidence above `0.6` every `0.5` do:

```shell
python3 main.py --dataset oak-dataset --api_key vkIkZac3CXvp0RZ31B3f --autoupload_threshold 0.6 --autoupload_interval 0.5
```

6. Experiment with `--upload_res` parameter. It allows for uploading annotated images in custom reosolution, which can
   be useful to create dataset with high-resolution images. For example, to upload in `2028x1520` do:

```
python3 main.py --dataset oak-dataset --api_key vkIkZac3CXvp0RZ31B3f --autoupload_threshold 0.6 --autoupload_interval 0.5 --upload_res 2028x1520
```