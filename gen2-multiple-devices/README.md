# Gen2 multiple devices per host

This example shows how you can use multiple DepthAI's on a single host. The demo will find all devices connected to the host and display an RGB preview from each of them. Note that at the end of the code we have to close the connection to all devices since we don't use context manager.

**Demo**
![image](https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/images/multiple-devices.png)

Just two DepthAI's looking at each other.

## Setup

```
python3 -m pip -U pip
python3 -m pip install -r requirements.txt
```

## Run

```
python3 main.py
```