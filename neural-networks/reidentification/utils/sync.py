import numpy as np
import depthai as dai


class AnnotationSyncNode2(dai.node.ThreadedHostNode):
    """A host node for syncing annotations in a two-stage pipeline.
    The node receives detections and recognitions from two different stages of the pipeline,
    merges them, and sends the merged annotations to the next stage of the pipeline.

    Attributes
    ----------
    input_detections : dai.Input
        The input message for the detections.
    input_recognitions : dai.Input
        The input message for the recognitions.
    output_detections : dai.Output
        The output message for the merged detections.
    csim : float
        The cosine similarity threshold used for merging.
    label_basename : str
        The basename of the labels (e.g., "person"). The labels will be in the format "basename_0", "basename_1", etc.
    """

    def __init__(self, csim: float = 0.5, label_basename: str = "person") -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.input_recognitions = self.createInput()
        self.output_detections = self.createOutput()

        self._cos_sim_threshold = csim
        self._label_basename = label_basename

        self._embeddings_dict = {}

    def run(self) -> None:

        while self.isRunning():
            dets = self.input_detections.get()

            for detection in dets.detections:
                rec = self.input_recognitions.get()
                detection.label_name = self._get_label(rec, self._label_basename)

            self.output_detections.send(dets)

    def _get_label(self, rec, basename) -> str:
        embedding = rec.getTensor("output", dequantize=True)
        sim = [
            self._cos_sim(embedding, self._embeddings_dict[key])
            for key in self._embeddings_dict
        ]

        if sim:
            if max(sim) >= self._cos_sim_threshold:
                idx = sim.index(max(sim))
                label = list(self._embeddings_dict.keys())[idx]
            else:
                label = f"{basename}_{len(self._embeddings_dict)}"
        else:
            label = f"{basename}_0"

        self._embeddings_dict[label] = embedding
        return label

    def _cos_sim(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y = Y.T
        result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
        return result.item()
