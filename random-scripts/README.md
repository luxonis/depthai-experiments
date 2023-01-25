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


### PoE script

[poe_test.py](poe_test.py) script connects to the device and tests PoE link speed, duplex mode, and boot mode.

```
py .\poe_latency.py
Connecting to  169.254.1.222 ...
mxid: 18443010617DC51200 (OK)
speed: 1000 (OK)
full duplex: 1 (OK)
boot mode: 3 (OK)
```
