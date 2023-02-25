# OAK PoE TCP streaming

This example shows how you can stream frames (and other data) with OAK PoE via TCP protocol from the Script node.

## Run this example

1. Run `cd ./api`.
2. Run `oak.py`, which will upload the pipeline to the OAK PoE. You will receive a log similar to `[192.168.1.66] [25.797] [Script(2)] [warning] Server up`.
3. Copy the IP of your OAK PoE camera (in my case `192.168.1.66`).
4. Edit the `OAK_IP` variable inside `host.py` with the IP of your OAK PoE camera.
5. Start `host.py` script, which will connect to the OAK PoE. Connection will be initialized and OAK PoE will start streaming encoded frames to the host computer.
6. Host computer will decode MJPEG encoded frames and show them to the user.

### OAK PoE as client

In this case, OAK PoE camera acts as a server, and host computer connects to it. You could also do it vice-verca, so the server would run on the host computer and OAK PoE would connect to it and start streaming frames. For that, see [PoE Client demo](poe-client).

## Install project requirements

```
cd ./api && python3 -m pip install -r requirements.txt
```
