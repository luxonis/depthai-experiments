# Gen2 Triangulation - Stereo neural inference demo

Because there are often application-specific host-side filtering to be done on the stereo
neural inference results, and because these calculations are lightweight
(i.e. could be done on an ESP32), we leave the triangulation itself to the host.

This 3D visualizer is for the facial landmarks demo, and uses OpenGL and OpenCV.
Consider it a draft/reference at this point.

## Demo

![Stereo Inference GIF](https://user-images.githubusercontent.com/59799831/132098832-70a2d0b9-1a30-4994-8dad-dc880a803fb3.gif)

## Installation

```
sudo apt-get install python3-pygame
python3 -m pip install -r requirements.txt
```

Note that this experiment uses the `Script` node that is currently in alpha mode, so you have to install the latest `gen2-scripting` branch of the library (you get it by installing `requirements.txt`)

## Usage

Run the application

```
python3 main.py
```

You should see 5 windows appear:
- `mono_left` which will show camera output from left mono camera + face bounding box & facial landmarks
- `mono_right` which will show camera output from right mono camera + face bounding box & facial landmarks
- `crop_left` which will show 48x48 left cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `crop_right` which will show 48x48 right cropped image that goes into the second NN + facial landmarsk that get outputed from the second NN
- `pygame window` which will show the triangulation results

## Troubleshooting

If you happen to see the error `/bin/sh: 1: sdl-config: not found` during command installation, in a log similar to this one
```
Collecting pygame==1.9.6
  Using cached pygame-1.9.6.tar.gz (3.2 MB)
    ERROR: Command errored out with exit status 1:
     command: /home/vandavv/dev/luxonis/venvs/develop/bin/python3 -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-clr4n0vn/pygame_2163392aa09745ef942ecb93ecf80adc/setup.py'"'"'; __file__='"'"'/tmp/pip-install-clr4n0vn/pygame_2163392aa09745ef942ecb93ecf80adc/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base /tmp/pip-pip-egg-info-a1i9ij3c
         cwd: /tmp/pip-install-clr4n0vn/pygame_2163392aa09745ef942ecb93ecf80adc/
    Complete output (12 lines):
    
    
    WARNING, No "Setup" File Exists, Running "buildconfig/config.py"
    Using UNIX configuration...
    
    /bin/sh: 1: sdl-config: not found
    /bin/sh: 1: sdl-config: not found
    /bin/sh: 1: sdl-config: not found
    
    Hunting dependencies...
    WARNING: "sdl-config" failed!
    Unable to run "sdl-config". Please make sure a development version of SDL is installed.
    ----------------------------------------
WARNING: Discarding https://files.pythonhosted.org/packages/0f/9c/78626be04e193c0624842090fe5555b3805c050dfaa81c8094d6441db2be/pygame-1.9.6.tar.gz#sha256=301c6428c0880ecd4a9e3951b80e539c33863b6ff356a443db1758de4f297957 (from https://pypi.org/simple/pygame/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
ERROR: Could not find a version that satisfies the requirement pygame==1.9.6 (from versions: 1.9.2rc1, 1.9.2, 1.9.3, 1.9.4.dev0, 1.9.4rc1, 1.9.4, 1.9.5rc1, 1.9.5rc2, 1.9.5, 1.9.6rc1, 1.9.6rc2, 1.9.6, 2.0.0.dev1, 2.0.0.dev2, 2.0.0.dev3, 2.0.0.dev4, 2.0.0.dev6, 2.0.0.dev8, 2.0.0.dev10, 2.0.0.dev12, 2.0.0.dev14, 2.0.0.dev16, 2.0.0.dev18, 2.0.0.dev20, 2.0.0.dev22, 2.0.0.dev24, 2.0.0, 2.0.1.dev1, 2.0.1)
ERROR: No matching distribution found for pygame==1.9.6
```

Please run the following command in order to fix the issue (per [this thread](https://stackoverflow.com/a/60990677/5494277))

```
$ sudo apt-get install python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
```