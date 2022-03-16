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

![alt text](https://github.com/ZigiMigi/images/blob/4f2b3fbe1ff1e9b6109eb15a72065bf2807391db/pic1.png)

And after that chose ``Foxglove websocket``.

![alt text](https://github.com/ZigiMigi/images/blob/4f2b3fbe1ff1e9b6109eb15a72065bf2807391db/pic2.png)

When you are successfully connected, chose ``Image`` panel and your studio should look something like this:

![alt text](https://github.com/ZigiMigi/images/blob/4f2b3fbe1ff1e9b6109eb15a72065bf2807391db/pic3.png)

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

![alt text](https://github.com/ZigiMigi/images/blob/4f2b3fbe1ff1e9b6109eb15a72065bf2807391db/pic4.png)
