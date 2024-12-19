## Foxglove

This example shows you how to use OAK camera for streaming to ``Foxglove Studio``. Example uses ``Foxglove``'s ``websocket`` to stream frames or point clouds.
The example only works on versions of python where the open3d library is supported.

# Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```

By default, the program will stream ``Color`` stream only. To enable other streams use next flags:

```
optional arguments:
  -h, --help            show this help message and exit
  -l, --left            enable left camera stream
  -r, --right           enable right camera stream
  -pc, --pointcloud     enable pointcloud stream
  -nc, --no-color       disable color camera stream
```

and open ``Foxglove Studio``. When you open up the studio you will be prompted with the connection type. Chose ``Open connection``

![pic1](https://user-ixmages.githubusercontent.com/82703447/161803788-3d0e15e9-df24-430b-8f73-4fdc82626c06.png)

And after that chose ``Foxglove websocket``.

![pic2](https://user-images.githubusercontent.com/82703447/161803642-a91e31af-18d0-4e53-babf-4268323e1255.png)

When you are successfully connected, chose ``Image`` panel and your studio should look something like this:

![pic3](https://user-images.githubusercontent.com/82703447/161803876-f3b168ed-4ca5-4059-84a5-daee22ae9db6.png)

Studio should now be displaying your ``point cloud``, if you open it up in the ``3D`` panel. You might need to toggle its visibility in the web app.

![pic4](https://user-images.githubusercontent.com/82703447/161804066-2f736ca3-07cd-413b-bb80-e8f71f2e53e7.png)
