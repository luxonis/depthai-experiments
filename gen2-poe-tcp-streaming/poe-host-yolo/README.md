# OAK PoE TCP streaming video + YOLOv8 object detection

Very similar to [OAK PoE TCP streaming](../), but this example also streams YOLO detection results to the host computer. The YOLO detection is done on the OAK PoE device, and the results are sent to the host computer where
they are displayed on the frame.

## Run this example

1. Run `oak.py`, which will upload the pipeline to the OAK PoE. You will receive a log similar to `[169.254.1.222] [25.797] [Script(2)] [warning] Server up`.
2. Copy the IP of your OAK PoE camera (in my case `169.254.1.222`).
3. Edit the `OAK_IP` variable inside `host.py` with the IP of your OAK PoE camera.
4. Start `host.py` script, which will connect to the OAK PoE. Connection will be initialized and OAK PoE will start streaming (unencoded) frames + YOLO inference results to the host computer.
5. Host computer will read frames and visualize YOLO results on these frames and show them to the user.