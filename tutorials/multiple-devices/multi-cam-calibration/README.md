# Multi camera calibration

This example demonstrates how to compute extrinsic parameters (pose of the camera) for multiple cameras.

The core extrinsic calibration uses OpenCV as follows:

1. **Image Capture**: A high-resolution still image featuring a checkerboard is captured.
1. **Corner Detection**:
   - [`cv2.findChessboardCorners`](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a) is used on the captured image to find initial locations of the inner corners of the visible checkerboard pattern.
   - [`cv2.cornerSubPix`](https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e) then refines these corner locations to sub-pixel accuracy.
1. **Pose Estimation ([`cv2.solvePnP`](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d))**:
   - This crucial function calculates the camera's pose. It takes:
     - The refined 2D pixel coordinates of the checkerboard corners found in the image.
     - The corresponding 3D real-world coordinates of these corners (which are known from the checkerboard's physical dimensions: `square_size` and `checkerboard_size`).
     - The camera's pre-existing intrinsic matrix `intrinsic_mat_still`.
     - The distortion coefficients.
   - It outputs the rotation vector (`rvec`) and translation vector (`tvec`), which together define the 3D pose of the checkerboard relative to the camera. These are the extrinsic parameters.
1. **Saving & Visualization**:
   - The computed pose (`rvec`, `tvec`) is saved.
   - For display, [`cv2.projectPoints`](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c) is used. It takes the 3D coordinates of the world axes (defined at the checkerboard's origin), the calculated `rvec` and `tvec`, and the camera's high-resolution intrinsic matrix to project these axes onto the 2D image plane. These projected 2D points are then scaled and normalized for drawing on the live preview.

## Demo

Program will also print USB speed, and connected cameras for each connected device before starting the pipeline. Output example for having connected OAK-D S2, OAK-D PRO

```
Found 2 DepthAI devices to configure.

Attempting to connect to device: 19443010F1E61F1300...
=== Successfully connected to device: 19443010F1E61F1300
    >>> Cameras: ['CAM_A', 'CAM_B', 'CAM_C']
    >>> USB speed: SUPER
    Pipeline created for device: 19443010F1E61F1300
    Pipeline for 19443010F1E61F1300 configured. Ready to be started.

Attempting to connect to device: 14442C1011D6C5D600...
=== Successfully connected to device: 14442C1011D6C5D600
    >>> Cameras: ['CAM_A', 'CAM_B', 'CAM_C']
    >>> USB speed: SUPER
    Pipeline created for device: 14442C1011D6C5D600
    Pipeline for 14442C1011D6C5D600 configured. Ready to be started.

```

## Usage

Running this example requires a **Luxonis device** connected to your computer. Refer to the [documentation](https://stg.docs.luxonis.com/software/) to setup your device if you haven't done it already.

You can run the experiment using your computer as host ([`PERIPHERAL` mode](#peripheral-mode)).

Here is a list of all available parameters:

```
--include-ip          Also include IP-only cameras (e.g. OAK-4) in the device list
--max-devices MAX_DEVICES
                        Limit the total number of devices to this count
```

### Controls

| Key | Action                                                                                                                                                                   |
| :-- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q` | **Quit**: Closes all camera connections and exits the application.                                                                                                       |
| `a` | **Camera Selection**: Cycles the calibration focus to the next available camera. The selected camera will be indicated by the on-screen text "Selected for Calib: True". |
| `c` | **Calibrate**: Captures a high-res still, uses a checkerboard to compute and save the camera's pose, then displays the world axes.                                       |

**Notes:**

- Ensure the checkerboard is clearly visible to the camera you intend to calibrate before pressing `c`.
- Ensure the checkerboard properties are correctly configured, currently it is configured for the checkerboard in this repo [checkerboard](pattern.pdf).
- Calibration results for each camera are saved in a file named `extrinsics_[MXID].npz`, where `[MXID]` is the unique ID of the DepthAI device.

### Peripheral Mode

Running in peripheral mode requires a host computer and there will be communication between device and host which could affect the overall speed of the app.
You can find more information about the supported devices and the set up instructions in our [Documentation](https://rvc4.docs.luxonis.com/hardware).
Moreover, you need to prepare a **Python 3.10** environment with the following packages installed:

- [DepthAI](https://pypi.org/project/depthai/)

You can simply install them by running:

```bash
pip install -r requirements.txt
```

#### Examples

```bash
python main.py
```

This will run the demo using only internal DepthAI cameras.

```bash
python main.py --include-ip
```

This will also discover and use any TCP/IP cameras on the network.

```bash
python main.py --max-devices 3
```

This will stop after configuring the first 3 devices.

```bash
python main.py --include-ip --max-devices 3
```

This will include IP cameras and then only use the first 3 discovered devices.
