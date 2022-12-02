import blobconverter
import depthai as dai

modelPath = blobconverter.from_zoo(name="mobilenet-ssd")

blob = dai.OpenVINO.Blob(modelPath)
print('Inputs')
[print(f"Name: {name}, Type: {vec.dataType}, Shape: {vec.dims} ({vec.order})") for name, vec in blob.networkInputs.items()]
print('Outputs')
[print(f"Name: {name}, Type: {vec.dataType}, Shape: {vec.dims} ({vec.order})") for name, vec in blob.networkOutputs.items()]
