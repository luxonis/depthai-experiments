from pathlib import Path
from typing import overload
import depthai as dai
import numpy as np


class BatchingNeuralNetwork(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self._nn = self.getParentPipeline().create(dai.node.NeuralNetwork)
        self.input = self.createInput()
        self.out = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.NNData, True)])
        self._input_shape = 4


    def build(self, max_size: int = 4, blocking: bool = False) -> "BatchingNeuralNetwork":
        self._nn_in_q = self._nn.input.createInputQueue()
        self._nn_out_q = self._nn.out.createOutputQueue(maxSize=max_size, blocking=blocking)
        return self
    

    def setInputShapeLen(self, shape: int) -> None:
        self._input_shape = shape


    def run(self):
        while self.isRunning():
            batched_nn_data: dai.NNData = self.input.get()
            input_layer_names = batched_nn_data.getAllLayerNames()
            if not input_layer_names:
                self.out.send(batched_nn_data)
                continue

            first_tensor: np.ndarray = batched_nn_data.getFirstTensor()
            if first_tensor.size == 0:
                self.out.send(batched_nn_data)
                continue

            data_shape = first_tensor.shape
            if len(data_shape) != self._input_shape:
                print(data_shape)
                raise ValueError(f"Expected {self._input_shape}D tensor, got {len(data_shape)}D tensor on the first layer. Set the correct input shape with set_input_shape() method. Neural network ID: {self._nn.id}")
            
            batch_size = data_shape[0]
            if batch_size == 0:
                self.out.send(batched_nn_data)
                continue

            result_nn_data: list[dai.NNData] = []
            for batch_index in range(batch_size):
                nn_data = self._get_nn_data(batched_nn_data, input_layer_names, batch_index)
                result_nn_data.append(nn_data)
            
            output_nn_data = self._merge_nn_data(result_nn_data)
            self._copy_sync_data(batched_nn_data, output_nn_data)
            self.out.send(output_nn_data)

    
    def _get_nn_data(self, batched_nn_data: dai.NNData, input_layer_names: list[str], batch_index: int):
        batch_nn_data = dai.NNData()
        for input_layer in input_layer_names:
            tensor: np.ndarray = batched_nn_data.getTensor(input_layer)
            batch_nn_data.addTensor(input_layer, tensor[batch_index])
        self._nn_in_q.send(batch_nn_data)
        nn_data: dai.NNData = self._nn_out_q.get()
        return nn_data


    def _merge_nn_data(self, result_nn_data: list[dai.NNData]) -> dai.NNData:
        output_nn_data = dai.NNData()
        first_data = result_nn_data[0]

        output_layer_names = first_data.getAllLayerNames()
        for output_layer in output_layer_names:
            output_tensor = np.concatenate([nn_data.getTensor(output_layer) for nn_data in result_nn_data])
            output_nn_data.addTensor(output_layer, output_tensor)
        return output_nn_data


    def _copy_sync_data(self, batched_nn_data: dai.NNData, output_nn_data: dai.NNData) -> None:
        output_nn_data.setSequenceNum(batched_nn_data.getSequenceNum())
        output_nn_data.setTimestamp(batched_nn_data.getTimestamp())
        output_nn_data.setTimestampDevice(batched_nn_data.getTimestampDevice())


    def getNumInferenceThreads(self) -> int:
        return self._nn.getNumInferenceThreads()
    

    def setBackend(self, setBackend: str) -> None:
        return self._nn.setBackend(setBackend)
    

    def setBackendProperties(self, setBackendProperties: dict[str,str]) -> None:
        return self._nn.setBackendProperties(setBackendProperties)
    

    @overload
    def setBlob(self, blob: dai.OpenVINO.Blob) -> None:
        return self._nn.setBlob(blob)
    

    @overload
    def setBlob(self, path: Path) -> None:
        return self._nn.setBlob(path)
    

    def setBlobPath(self, path: Path) -> None:
        return self._nn.setBlobPath(path)
    

    def setNumInferenceThreads(self, numThreads: int) -> None:
        return self._nn.setNumInferenceThreads(numThreads)
    

    def setNumNCEPerInferenceThread(self, numNCEPerThread: int) -> None:
        return self._nn.setNumNCEPerInferenceThread(numNCEPerThread)
    

    def setNumPoolFrames(self, numFrames: int) -> None:
        return self._nn.setNumPoolFrames(numFrames)
    

    def setNumShavesPerInferenceThread(self, numShavesPerInferenceThread: int) -> None:
        return self._nn.setNumShavesPerInferenceThread(numShavesPerInferenceThread)
    

    def setXmlModelPath(self, xmlModelPath: Path, binModelPath: Path = ...) -> None:
        return self._nn.setXmlModelPath(xmlModelPath, binModelPath)
    
    
    @property
    def id(self) -> int:
        return self._nn.id