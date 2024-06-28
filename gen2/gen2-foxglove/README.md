## Gen2 foxglove

This example shows you how to use OAK camera for streaming to ``Foxglove Studio``. Example uses ``Foxglove``'s ``websocket`` to stream frames or point clouds.
Before beginning, you will have to run next command:
```bash
python3 requrements.txt
```

After that, run program with:

```bash
python3 main.py
```

and open ``Foxglove Studio``. When you open up the studio you will be prompted with the connection type. Chose ``Open connection``

![pic1](https://user-images.githubusercontent.com/82703447/161803788-3d0e15e9-df24-430b-8f73-4fdc82626c06.png)

And after that chose ``Foxglove websocket``.

![pic2](https://user-images.githubusercontent.com/82703447/161803642-a91e31af-18d0-4e53-babf-4268323e1255.png)

When you are successfully connected, chose ``Image`` panel and your studio should look something like this:

![pic3](https://user-images.githubusercontent.com/82703447/161803876-f3b168ed-4ca5-4059-84a5-daee22ae9db6.png)

By default, the program will stream ``Color`` stream only. To enable other streams use next flags:

 - ``-l`` for ``Left`` camera stream,
 - ``-r`` for ``Right`` camera stream,
 - ``-pcl`` for ``Point cloud`` streaming,
 - and, if you wish to disable ``Color`` stream use ``-c``.

Running ``Point cloud`` and ``Color`` streams are used like this:

```bash
python3 main.py -pcl
```

Studio should now be displaying your ``point cloud``, if you open it up in the ``3D`` panel.

![pic4](https://user-images.githubusercontent.com/82703447/161804066-2f736ca3-07cd-413b-bb80-e8f71f2e53e7.png)
