import numpy as np
import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class AnnotationSyncNode(dai.node.ThreadedHostNode):
    def __init__(self, csim: float = 0.7) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.input_recognitions = self.createInput()
        self.output_detections = self.createOutput()

        self._cos_sim_threshold = csim  # cosine similarity threshold
        self._labels_dict = {}  # label: embedding of the last seen entry

    def run(self) -> None:

        seq = 0  # sequence number
        labels = []

        while self.isRunning():
            rec = self.input_recognitions.get()
            rec_seq = rec.getSequenceNum()
            print("REC", rec_seq)

            if rec_seq > seq: # current rec is from a new sequence
                dets = self.input_detections.get()
                dets_seq = dets.getSequenceNum()
                print("DETS", dets_seq)

                if dets_seq < seq: # catch-up with detections if there is no rec
                    self._send_output(dets, [])
                    dets = self.input_detections.get()
                    dets_seq = dets.getSequenceNum()
                    print("DETS", dets_seq)

                if dets_seq == seq:
                    self._send_output(dets, labels)
                    labels = []
                    seq = rec_seq
                else:
                    raise Exception(f"SequenceNum mismatch.")

            labels.append(self._get_label(rec))

    def _get_label(self, rec, base="person") -> str:
        embedding = rec.getTensor("output", dequantize=True)
        sim = [
            self._cos_sim(embedding, self._labels_dict[key])
            for key in self._labels_dict
        ]

        if sim:
            if max(sim) >= self._cos_sim_threshold:
                idx = sim.index(max(sim))
                label = list(self._labels_dict.keys())[idx]
            else:
                label = f"{base}_{len(self._labels_dict)}"
        else:
            label = f"{base}_0"

        self._labels_dict[label] = embedding
        return label

    def _cos_sim(self, X: np.ndarray, Y: np.ndarray) -> float:
        Y = Y.T
        result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
        return result.item()

    def _send_output(self, dets: ImgDetectionsExtended, labels: list[str]) -> None:
        if len(labels) != len(dets.detections):
            raise ValueError("Labels and detections must have the same length")

        for i, detection in enumerate(dets.detections):
            detection.label_name = labels[i]

        self.output_detections.send(dets)
