# OAK PoE TCP streaming - Configure focus

Very similar to [OAK PoE TCP streaming](../), this example only adds option to configure OAK PoE's focus from the host with `.` and `,` keys.

## Run this example

1. Run `oak.py`, which will upload the pipeline to the OAK PoE. You will receive a log similar to `[169.254.1.222] [25.797] [Script(2)] [warning] Server up`.
2. Copy the IP of your OAK PoE camera (in my case `169.254.1.222`).
3. Edit the `OAK_IP` variable inside `host.py` with the IP of your OAK PoE camera.
4. Start `host.py` script, which will connect to the OAK PoE. Connection will be initialized and OAK PoE will start streaming encoded frames to the host computer.
5. Host computer will decode MJPEG encoded frames and show them to the user.
6. You can control OAK PoE's focus from the host by pressing `.` and `,` keys on the frame. This will send TCP packet to the OAK PoE which will parse the packet and send CameraControl msg to the ColorCamera node.