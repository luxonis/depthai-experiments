# Box Measurement

![box](https://github.com/luxonis/depthai-experiments/assets/18037362/e6657b48-0f10-4335-8491-47bae9b8ade5)

This experiment was designed for OAK-D-SR-POE (with ToF sensor). If you're using stereo camera, please see [Box Measurement API demo](./api/).

## How to run

1. **Roboflow**

    This demo uses [cardboard-box AI model](https://app.roboflow.com/are-bla-gpxot/cardboard-box-u35qd/1) trained with [Roboflow](https://roboflow.com). You can create a free account, head over to `API Keys`, generate a new Private API key, and enter it inside the `model_config.key` field.
2. **Requirements**

    (Optionally use venv)
    ```bash
    python3 -m pip install -r requirements.txt
    ```

3. **Run & Calibrate**

    The app first needs to estimate the ground plane on which the box is placed. To do this, most of the camera's view should be the flat surface. Once this is achieved, input `c` into the terminal, and the app will calibrate the ground plane. This will generate a new file `plane_eq.json` which will have the plane equation (4 values). You only need to do this once, not on every run. If you move the camera, you will need to recalibrate. Plane is visualized with a green square in the pointcloud window.

4. **Box measurement**

    After successful calibration, the app will start detecting the box and measuring its dimensions. The dimensions are displayed directly inside the viewer window.

