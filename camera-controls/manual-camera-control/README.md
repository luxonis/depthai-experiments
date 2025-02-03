# Manual Camera Control

This experiment demonstrates how to manually control different camera parameters. Use keyboard to modify different settings.

## Demo

TODO

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Usage

You can run the experiment in fully on device (`STANDALONE` mode) or using your computer as host (`PERIPHERAL` mode).

### Peripheral Mode

```bash
python3 main.py --device <DEVICE> --fps_limit <FPS_LIMIT>
```

- `<DEVICE>`: Device IP or ID. Default: \`\`.
- `<FPS_LIMIT>`: Limit of the camera FPS. Default: `30`.

#### Examples

```bash
python3 main.py
```

This will run the manual camera control experiment with the default device and camera input.

### Standalone Mode

TODO

### Keyboard Controls

| Key      | Description                          |
| -------- | ------------------------------------ |
| `c`      | Capture an image                     |
| `e`      | Autoexposure                         |
| `t`      | Trigger autofocus                    |
| `f`      | Autofocus (continuous)               |
| `w`      | Auto white balance lock (true/false) |
| `r`      | Auto exposure lock (true/false)      |
| `+`, `-` | Increase/decrease selected control   |

The following controls can be selected and modified with `+` and `-` keys:

| Key | Description                |
| --- | -------------------------- |
| `1` | Manual exposure time       |
| `2` | Manual sensitivity ISO     |
| `3` | Auto white balance mode    |
| `4` | Auto exposure compensation |
| `5` | Anti-banding/flicker mode  |
| `6` | Effect mode                |
| `7` | Brightness                 |
| `8` | Contrast                   |
| `9` | Saturation                 |
| `0` | Sharpness                  |
| `o` | Manual white balance       |
| `p` | Manual focus               |
| `[` | Luma denoise               |
| `]` | Chroma denoise             |
