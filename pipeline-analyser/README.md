# DepthAI Pipeline Graph

For **debugging purposes we suggest using [DepthAI Pipeline Graph](https://github.com/geaxgx/depthai_pipeline_graph) visualizing tool**, as it's more sophisticated and provides **better UX** than this simple demo.


## Pipeline JSON file analyser

This experiment prints relevant information about JSON-serialized pipeline. Useful for debug purposes when analysing the pipeline

## Usage

Script works without any extra dependencies.

Navigate to `api` folder:

```bash
cd ./api
```

Then, run the following command:

```
python3 analyse.py <path_to_pipeline_json_file>
```

## Demo

To see an example of the visualization, you can use `example_pipeline.json` file:

```
$ python3 analyse.py example_pipeline.json

=== LEVEL 0 ====
XLinkIn
	Connections: "out" to MonoCamera "inputControl"
	Properties: [185, 3, 189, 12, 108, 101, 102, 116, 95, 99, 111, 110, 116, 114, 111, 108, 129, 0, 4, 8]

XLinkIn
	Connections: "out" to MonoCamera "inputControl"
	Properties: [185, 3, 189, 13, 114, 105, 103, 104, 116, 95, 99, 111, 110, 116, 114, 111, 108, 129, 0, 4, 8]

XLinkIn
	Connections: "out" to ColorCamera "inputControl"
	Properties: [185, 3, 189, 13, 99, 111, 108, 111, 114, 95, 99, 111, 110, 116, 114, 111, 108, 129, 0, 4, 8]

XLinkIn
	Connections: "out" to StereoDepth "inputConfig"
	Properties: [185, 3, 189, 12, 115, 116, 101, 114, 101, 111, 67, 111, 110, 102, 105, 103, 129, 0, 4, 8]

=== LEVEL 1 ====
MonoCamera
	Connections: "out" to StereoDepth "left" and "out" to VideoEncoder "in"
	Properties: [185, 5, 185, 20, 0, 3, 0, 185, 3, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 255, 2, 136, 0, 0, 240, 65]

MonoCamera
	Connections: "out" to VideoEncoder "in" and "out" to StereoDepth "right"
	Properties: [185, 5, 185, 20, 0, 3, 0, 185, 3, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 255, 2, 136, 0, 0, 240, 65]

ColorCamera
	Connections: "preview" to ImageManip "inputImage" and "video" to VideoEncoder "in"
	Properties: [185, 18, 185, 20, 0, 3, 0, 185, 3, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 185, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 129, 64, 1, 129, 64, 2, 133, 128, 7, 133, 56, 4, 255, 255, 0, 136, 0, 0, 240, 65, 136, 0, 0, 128, 191, 136, 0, 0, 128, 191, 0, 185, 4, 0, 0, 0, 0]

=== LEVEL 2 ====
VideoEncoder
	Connections: "bitstream" to XLinkOut "in"
	Properties: [185, 10, 0, 30, 0, 0, 0, 4, 50, 0, 0, 136, 0, 0, 240, 65]

VideoEncoder
	Connections: "bitstream" to XLinkOut "in"
	Properties: [185, 10, 0, 30, 0, 0, 0, 4, 50, 0, 0, 136, 0, 0, 240, 65]

VideoEncoder
	Connections: "bitstream" to XLinkOut "in"
	Properties: [185, 10, 0, 30, 0, 0, 0, 4, 50, 0, 0, 136, 0, 0, 240, 65]

StereoDepth
	Connections: "depth" to SpatialDetectionNetwork "inputDepth" and "rectifiedRight" to XLinkOut "in" and "rectifiedLeft" to XLinkOut "in" and "depth" to XLinkOut "in"
	Properties: [185, 14, 185, 5, 185, 6, 0, 0, 0, 0, 4, 3, 185, 7, 7, 0, 185, 5, 0, 2, 136, 0, 0, 0, 63, 0, 1, 185, 4, 0, 3, 136, 205, 204, 204, 62, 0, 185, 2, 0, 134, 255, 255, 0, 0, 185, 2, 0, 50, 185, 2, 1, 0, 185, 4, 255, 0, 1, 0, 185, 5, 1, 0, 0, 128, 245, 185, 3, 0, 2, 127, 185, 5, 1, 128, 250, 129, 244, 1, 128, 250, 129, 244, 1, 255, 1, 255, 190, 190, 190, 190, 1, 185, 5, 189, 0, 189, 0, 190, 16, 16, 1, 3, 255, 255]

ImageManip
	Connections: "out" to SpatialDetectionNetwork "in"
	Properties: [185, 3, 185, 8, 185, 7, 185, 4, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 185, 3, 185, 2, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 185, 2, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 136, 0, 0, 0, 0, 0, 136, 0, 0, 128, 63, 136, 0, 0, 128, 63, 0, 1, 185, 15, 133, 44, 1, 133, 44, 1, 0, 0, 0, 0, 186, 0, 1, 0, 186, 0, 0, 0, 136, 0, 0, 0, 0, 0, 0, 185, 2, 8, 0, 0, 1, 1, 0, 0, 134, 0, 0, 16, 0, 4]

=== LEVEL 3 ====
XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 4, 108, 101, 102, 116, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 5, 114, 105, 103, 104, 116, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 5, 99, 111, 108, 111, 114, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 8, 100, 101, 112, 116, 104, 82, 97, 119, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 13, 114, 101, 99, 116, 105, 102, 105, 101, 100, 76, 101, 102, 116, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 14, 114, 101, 99, 116, 105, 102, 105, 101, 100, 82, 105, 103, 104, 116, 0]

SpatialDetectionNetwork
	Connections: "passthrough" to XLinkOut "in" and "out" to XLinkOut "in"
	Properties: [185, 15, 1, 130, 0, 137, 221, 0, 189, 12, 97, 115, 115, 101, 116, 58, 95, 95, 98, 108, 111, 98, 8, 2, 0, 136, 0, 0, 0, 63, 0, 0, 186, 0, 187, 0, 136, 0, 0, 0, 0, 136, 154, 153, 153, 62, 185, 2, 100, 129, 16, 39, 0]

=== LEVEL 4 ====
XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 5, 110, 110, 79, 117, 116, 0]

XLinkOut
	No connections
	Properties: [185, 3, 136, 0, 0, 128, 191, 189, 7, 110, 110, 73, 110, 112, 117, 116, 0]
```
