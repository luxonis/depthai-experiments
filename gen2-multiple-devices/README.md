# Gen2 multiple devices per host

This example shows how you can use multiple DepthAI's on a single host. Note that at the end of the code we close the connection to all devices since we don't use context manager.

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