import cv2
import depthai as dai
import numpy as np
from sklearn.cluster import DBSCAN

# Set max clusters for color output (number of lines + a few more due to errors in fast postprocessing)
MAX_CLUSTERS = 6


class LaneNet(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self, preview: dai.Node.Output, nn: dai.Node.Output, nn_shape: tuple[int, int]
    ) -> "LaneNet":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self.nn_shape = nn_shape
        return self

    # preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, nn: dai.NNData) -> None:
        assert isinstance(preview, dai.ImgFrame)
        frame = preview.getCvFrame()

        # First layer is a binary segmentation mask of lanes
        segmentation = (
            nn.getTensor("LaneNet/bisenetv2_backend/binary_seg/ArgMax/Squeeze")
            .flatten()
            .astype(np.uint8)
        )
        # Second layer is an array of embeddings of dimension 4 for each pixel in the input
        embeddings = nn.getTensor(
            "LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/Conv2D"
        )
        embeddings = embeddings.reshape((4, self.nn_shape[0] * self.nn_shape[1]))

        # Cluster outputs
        clusters = self.cluster_outputs(segmentation, embeddings)

        overlay = create_overlay(clusters)
        frame = cv2.addWeighted(frame, 1, overlay, 0.8, 0)

        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord("q"):
            print("Pipeline exited.")
            self.stopPipeline()

    # Perform postprocessing - clustering of line embeddings using DBSCAN
    #  note that this is not whole postprocessing, just a quick implementation to show what is possible
    #  if you want to use whole postprocessing please refer to the LaneNet paper: https://arxiv.org/abs/1802.05591
    def cluster_outputs(self, binary_seg_ret, instance_seg_ret):
        # Create mask from binary output
        mask = binary_seg_ret.copy()
        mask = mask.astype(bool)

        # Mask out embeddings
        embeddings = instance_seg_ret.copy()
        embeddings = np.transpose(embeddings, (1, 0))
        embeddings_masked = embeddings[mask]

        # Sort so same classes are sorted first each time and generate inverse sort
        #  works only if new lanes are added on the right side
        idx = np.lexsort(np.transpose(embeddings_masked)[::-1])
        idx_inverse = np.empty_like(idx)
        idx_inverse[idx] = np.arange(idx.size)
        embeddings_masked = embeddings_masked[idx]

        # Cluster embeddings with DBSCAN
        clustering = DBSCAN(eps=0.4, min_samples=500, algorithm="kd_tree").fit(
            embeddings_masked
        )

        # Unsort so pixels match their positions again
        clustering_labels = clustering.labels_[idx_inverse]

        # Create an array of masked clusters
        clusters = np.zeros((self.nn_shape[0] * self.nn_shape[1],))
        clusters[mask] = clustering_labels + 1
        clusters = clusters.reshape(self.nn_shape[1], self.nn_shape[0])

        return clusters


def create_overlay(cluster_outputs):
    # Multiply to get classes between 0 and 255
    output = np.array(cluster_outputs) * (255 / MAX_CLUSTERS)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0, 0, 0]
    return output_colors
