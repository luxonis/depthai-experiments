# Depth people counting

This example demonstrates how to perform people counting from depth frames along a passageway. This approach could be useful for the privacy concerns of people counting applications.

This demo contains many hard-coded values specific to [depth-people-counting-01](api/depth-people-counting-01) depthai recording.If you would wish to use this application in your own setup, these values would higly depend on the OAK camera installation, its FOV, and the passageway structure.

## Demo

[![Depth people counting](https://user-images.githubusercontent.com/18037362/179425724-fcc77aa7-6616-4ca7-8083-ec1a7a78a7de.gif)](https://youtu.be/9M1mRICVKcw "Depth people counting")

## Usage

### Navigate to directory

```bash
cd ./api
```

### Pre-requisites

```bash
python3 -m pip install -r requirements.txt
```

### Launch the script

```bash
python3 main.py [--path PATH_TO_DEPTHAI_RECORDING]
```

> You can record your own DepthAI recording with [DepthAI record tool](../record-replay/).
