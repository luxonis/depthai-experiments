# OAK PoE TCP streaming

This example shows how you can stream frames (and other data) with OAK PoE via TCP protocol from the Script node.

## Run this example

1. Get IP of you computer (`ifconfig` or `ipconfig`). Let's say your computer's IP is `192.168.1.55`
2. Run `host.py` script. This will start the TCP server to which OAK PoE camera will connect to.
3. In `oak.py`, edit `HOST_IP` variable with the IP of the host.
4. Start `oak.py` script, which will connect to the host computer. Connection will be initialized and OAK PoE will start streaming encoded frames to the host computer.
5. Host computer will decode MJPEG encoded frames and show them to the user.

### OAK PoE as server

In this case, host computer acts as a server, and OAK PoE camera connects to it. You could also do it vice-verca, so the server would run on the OAK PoE camera and host computer would connect to it. For that, see [PoE Host demo](../..).

