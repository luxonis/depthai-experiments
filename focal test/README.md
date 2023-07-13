# Utilities

This folder contains DepthAI utility tools.

## Device Manager

![Device Manager](https://user-images.githubusercontent.com/18037362/171629704-0f78f31a-1778-4338-8ac0-bdfb0d2d593f.png)

``device_manager.py`` helps interfacing with the device [Bootloader](https://docs.luxonis.com/projects/api/en/latest/components/bootloader) and bootloader configuration. See [Device Manager Usage](https://docs.luxonis.com/projects/api/en/latest/components/bootloader/#device-manager-usage) to see how to use this utility.

## Standalone executable

Requirements:
```
# Linux/macOS
python3 -m pip install pyinstaller
# Windows
python -m pip install pyinstaller
```

To build standalone executable issue the following command:
```sh
pyinstaller --onefile -w --icon=assets/icon.ico --add-data="assets/icon.ico;assets" --add-data="assets/icon.png;assets" device_manager.py
```
