streaming depthai.py video via web browser localhost port 8090

Steps to use:
1. Download the file `depthai.py`
2. Replace with the current `depthai.py` under `depthai`
3. Run the file with commend `python3 test.py`
4. Open chrome with address: http://localhost:8090 and now you can see the video stream via web browser (Chrome)

The video below shows this code in action on macOS:
[![MJPEG Streaming DepthAI](https://img.youtube.com/vi/695o0EO1Daw/0.jpg)](https://www.youtube.com/watch?v=695o0EO1Daw "DepthAI on Mac")

Note: if want to view the video stream on different device under same WIFI, please fill your IP address to replace `localhost` at `server_HTTP = ThreadedHTTPServer(('localhost', 8090), VideoStreamHandler)` in the `depthai.py` 
