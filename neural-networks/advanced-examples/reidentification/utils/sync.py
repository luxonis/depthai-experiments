import numpy as np
import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class AnnotationSyncNode(dai.node.ThreadedHostNode):
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

    def __init__(self, csim: float = 0.7, label_basename: str = "person") -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.input_recognitions = self.createInput()
        self.output_detections = self.createOutput()

        self._cos_sim_threshold = csim
        self._label_basename = label_basename
        self._embeddings_dict = {}

    def run(self) -> None:
        seqs = set()
        labels = []

        while self.isRunning():
            if len(seqs) <= 1:  # at most one sequence stored
                rec = self.input_recognitions.get()
                seqs.add(rec.getSequenceNum())
                print("rec", seqs)

            if len(seqs) > 1:  # multiple sequences stored
                dets = self.input_detections.get()
                det_seq = dets.getSequenceNum()
                print("det_seq", det_seq)
                if det_seq < min(seqs):
                    self._send_output(dets, [])
                elif det_seq == min(seqs):
                    self._send_output(dets, labels)
                    seqs.remove(min(seqs))  # remove sequence
                    labels = []  # remove sequence labels
                else:
                    raise ValueError("Detections ahead of recognitions.")

            if len(seqs) <= 1:  # store labels of at most one sequence
                labels.append(self._get_label(rec, self._label_basename))

    def _get_label(self, rec, basename="person") -> str:
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

    def _send_output(self, dets: ImgDetectionsExtended, labels: list[str]) -> None:
        if len(labels) != len(dets.detections):
            raise ValueError("Detections and labels do not match.")

        for i, detection in enumerate(dets.detections):
            detection.label_name = labels[i]

        self.output_detections.send(dets)
