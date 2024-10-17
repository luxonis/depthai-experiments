# Box Measurement

![box](https://github.com/luxonis/depthai-experiments/assets/18037362/e6657b48-0f10-4335-8491-47bae9b8ade5)

This experiment was designed for OAK-D-SR-POE (with ToF sensor).

## How to run

1. **Roboflow**

    This demo uses [cardboard-box AI model](https://app.roboflow.com/are-bla-gpxot/cardboard-box-u35qd/1) trained with [Roboflow](https://roboflow.com). You can create a free account, head over to `API Keys`, generate a new Private API key, and enter it inside the `model_config.key` field.
2. **Requirements**

    (Optionally use venv)
    ```bash
    python3 -m pip install -r requirements.txt
    ```

3. **Run**

    The app automatically estimates the ground plane where the box is placed. Ensure that the majority of the camera's view captures the flat surface. The ground plane is visualized as a green square in the point cloud window, and this process runs dynamically, so there's no need for manual calibration. 

    For multi box segmentation run 
        ```bash
    python3 main_multi.py
    ```

    For single box segmentation run 
        ```bash
    python3 main_single.py
    ```

    For ground plane estimator only run 
        ```bash
    python3 main_ground.py
    ```

4. **Box measurement**

    The app will start detecting the box and measuring its dimensions. The dimensions are displayed directly inside the viewer window.

