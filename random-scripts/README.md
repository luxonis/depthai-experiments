# Scripts

Simple scripts that some people might find useful.

### Get blob input/output layers

[blob_inputs_ouputs.py](blob_inputs_ouputs.py) prints `.blob`s input/output and layers; their names, datatypes, shapes and channel orders.

```
py .\blob_inputs_ouputs.py
Inputs
Name: data, Type: DataType.U8F, Shape: [300, 300, 3, 1] (StorageOrder.NCHW)
Outputs
Name: detection_out, Type: DataType.FP16, Shape: [7, 100, 1, 1] (StorageOrder.NCHW)
```

### Get MxId

[get_mxid.py](get_mxid.py) gets the MxId (unique ID) of the (first) connected device.

### OAK bandwidth test

Sends large buffers to/from device to calculate uplink and downlink bandwidth. Results will vary a bit even at the same setup, and will especially vary
at different network configurations.

```
USB3 (5gbps, direct link):
Downlink 2273.6 mbps
Uplink 1070.6 mbps

POE (1gbps, direct link):
Downlink 805.8 mbps
Uplink 211.7 mbps
```

### OAK latency test

Calculates the roundabout latency (via loopback XLinkIn -> XLinkOut). Latency for the first frame will be about 50ms for both USB/POE devices, as
device/pipeline is still booting up.

```
USB3 (5gbps, direct link), latency is stable:
Average latency 1.00 ms

POE (1gbps, direct link), latency varies  quite a bit (from 2ms to 30ms):
Average latency 9.89 ms
```


### PoE test script

[poe_test.py](poe_test.py) script connects to the device and tests PoE link speed, duplex mode, and boot mode.

```
py .\poe_latency.py
Connecting to  169.254.1.222 ...
mxid: 18443010617DC51200 (OK)
speed: 1000 (OK)
full duplex: 1 (OK)
boot mode: 3 (OK)
```
